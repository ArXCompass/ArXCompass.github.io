# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10830v3">The Siren Song of LLMs: How Users Perceive and Respond to Dark Patterns in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 23 pages, 7 figures. Accepted at CHI 2026 (ACM Conference on Human Factors in Computing Systems), Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Large language models can influence users through conversation, creating new forms of dark patterns that differ from traditional UX dark patterns. We define LLM dark patterns as manipulative or deceptive behaviors enacted in dialogue. Drawing on prior work and AI incident reports, we outline a diverse set of categories with real-world examples. Using them, we conducted a scenario-based study where participants (N=34) compared manipulative and neutral LLM responses. Our results reveal that recognition of LLM dark patterns often hinged on conversational cues such as exaggerated agreement, biased framing, or privacy intrusions, but these behaviors were also sometimes normalized as ordinary assistance. Users' perceptions of these dark patterns shaped how they respond to them. Responsibilities for these behaviors were also attributed in different ways, with participants assigning it to companies and developers, the model itself, or to users. We conclude with implications for design, advocacy, and governance to safeguard user autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23001v3">Bias Beyond Borders: Political Ideology Evaluation and Steering in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 PrePrint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape global discourse, making fairness and ideological neutrality essential for responsible AI deployment. Despite growing attention to political bias in LLMs, prior work largely focuses on high-resource, Western languages or narrow multilingual settings, leaving cross-lingual consistency and safe post-hoc mitigation underexplored. To address this gap, we present a large-scale multilingual evaluation of political bias spanning 50 countries and 33 languages. We introduce a complementary post-hoc mitigation framework, Cross-Lingual Alignment Steering (CLAS), designed to augment existing steering methods by aligning ideological representations across languages and dynamically regulating intervention strength. This method aligns latent ideological representations induced by political prompts into a shared ideological subspace, ensuring cross lingual consistency, with the adaptive mechanism prevents over correction and preserves coherence. Experiments demonstrate substantial bias reduction along both economic and social axes with minimal degradation in response quality. The proposed framework establishes a scalable and interpretable paradigm for fairness-aware multilingual LLM governance, balancing ideological neutrality with linguistic and cultural diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16814v5">From Belief Entrenchment to Robust Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted to TACL
    </div>
    <details class="paper-abstract">
      Multi-Agent Debate (MAD) has emerged as a promising inference scaling method for Large Language Model (LLM) reasoning. However, it frequently suffers from belief entrenchment, where agents reinforce shared errors rather than correcting them. Going beyond merely identifying this failure, we decompose it into two distinct root causes: (1) the model's biased $\textit{static initial belief}$ and (2) $\textit{homogenized debate dynamics}$ that amplify the majority view regardless of correctness. To address these sequentially, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$. Our framework first rectifies the static belief via strategic prior knowledge elicitation, then reshapes the debate dynamics by enforcing perspective diversity. Validated on our new $\textit{MetaNIM Arena}$ benchmark, $\textbf{DReaMAD}$ significantly mitigates entrenchment, achieving a +9.5\% accuracy gain over ReAct prompting and a +19.0\% higher win rate than standard MAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10471v1">TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Given that Large Language Models (LLMs) are increasingly applied to automate software development, comprehensive software assurance spans three distinct goals: regression prevention, reactive reproduction, and proactive discovery. Current evaluations systematically overlook the third goal. Specifically, they either treat existing code as ground truth (a compliance trap) for regression prevention, or depend on post-failure artifacts (e.g., issue reports) for bug reproduction-so they rarely surface defects before failures. To bridge this gap, we present TestExplora, a benchmark designed to evaluate LLMs as proactive testers within full-scale, realistic repository environments. TestExplora contains 2,389 tasks from 482 repositories and hides all defect-related signals. Models must proactively find bugs by comparing implementations against documentation-derived intent, using documentation as the oracle. Furthermore, to keep evaluation sustainable and reduce leakage, we propose continuous, time-aware data collection. Our evaluation reveals a significant capability gap: state-of-the-art models achieve a maximum Fail-to-Pass (F2P) rate of only 16.06%. Further analysis indicates that navigating complex cross-module interactions and leveraging agentic exploration are critical to advancing LLMs toward autonomous software quality assurance. Consistent with this, SWEAgent instantiated with GPT-5-mini achieves an F2P of 17.27% and an F2P@5 of 29.7%, highlighting the effectiveness and promise of agentic exploration in proactive bug discovery tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.01076v4">When Speculation Spills Secrets: Side Channels via Speculative Decoding In LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Deployed large language models (LLMs) often rely on speculative decoding, a technique that generates and verifies multiple candidate tokens in parallel, to improve throughput and latency. In this work, we reveal a new side-channel whereby input-dependent patterns of correct and incorrect speculations can be inferred by monitoring per-iteration token counts or packet sizes. In evaluations using research prototypes and production-grade vLLM serving frameworks, we show that an adversary monitoring these patterns can fingerprint user queries (from a set of 50 prompts) with over 75% accuracy across four speculative-decoding schemes at temperature 0.3: REST (100%), LADE (91.6%), BiLD (95.2%), and EAGLE (77.6%). Even at temperature 1.0, accuracy remains far above the 2% random baseline - REST (99.6%), LADE (61.2%), BiLD (63.6%), and EAGLE (24%). We also show the capability of the attacker to leak confidential datastore contents used for prediction at rates exceeding 25 tokens/sec. To defend against these, we propose and evaluate a suite of mitigations, including packet padding and iteration-wise token aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10454v1">LATA: A Tool for LLM-Assisted Translation Annotation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The construction of high-quality parallel corpora for translation research has increasingly evolved from simple sentence alignment to complex, multi-layered annotation tasks. This methodological shift presents significant challenges for structurally divergent language pairs, such as Arabic--English, where standard automated tools frequently fail to capture deep linguistic shifts or semantic nuances. This paper introduces a novel, LLM-assisted interactive tool designed to reduce the gap between scalable automation and the rigorous precision required for expert human judgment. Unlike traditional statistical aligners, our system employs a template-based Prompt Manager that leverages large language models (LLMs) for sentence segmentation and alignment under strict JSON output constraints. In this tool, automated preprocessing integrates into a human-in-the-loop workflow, allowing researchers to refine alignments and apply custom translation technique annotations through a stand-off architecture. By leveraging LLM-assisted processing, the tool balances annotation efficiency with the linguistic precision required to analyze complex translation phenomena in specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10453v1">The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Models (LLMs) has resulted in a paradigm shift towards autonomous agents, necessitating robust security against Prompt Injection (PI) vulnerabilities where untrusted inputs hijack agent behaviors. This SoK presents a comprehensive overview of the PI landscape, covering attacks, defenses, and their evaluation practices. Through a systematic literature review and quantitative analysis, we establish taxonomies that categorize PI attacks by payload generation strategies (heuristic vs. optimization) and defenses by intervention stages (text, model, and execution levels). Our analysis reveals a key limitation shared by many existing defenses and benchmarks: they largely overlook context-dependent tasks, in which agents are authorized to rely on runtime environmental observations to determine actions. To address this gap, we introduce AgentPI, a new benchmark designed to systematically evaluate agent behavior under context-dependent interaction settings. Using AgentPI, we empirically evaluate representative defenses and show that no single approach can simultaneously achieve high trustworthiness, high utility, and low latency. Moreover, we show that many defenses appear effective under existing benchmarks by suppressing contextual inputs, yet fail to generalize to realistic agent settings where context-dependent reasoning is essential. This SoK distills key takeaways and open research problems, offering structured guidance for future research and practical deployment of secure LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09379v2">LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at https://github.com/Lingxi-mental-health/LingxiDiagBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11209v1">SAFuzz: Semantic-Guided Adaptive Fuzzing for LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 11 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      While AI-coding assistants accelerate software development, current testing frameworks struggle to keep pace with the resulting volume of AI-generated code. Traditional fuzzing techniques often allocate resources uniformly and lack semantic awareness of algorithmic vulnerability patterns, leading to inefficient resource usage and missed vulnerabilities. To address these limitations, we present a hybrid testing framework that leverages LLM-guided adaptive fuzzing to detect algorithmic vulnerabilities efficiently. Our system SAFuzz integrates prompt-based behavioral diversification, harness generation with problem-specific oracles, and an LLM-based predictor to enable adaptive resource allocation and dynamic early stopping. Evaluating SAFuzz on CSES algorithmic problems, we improve vulnerability discrimination precision from 77.9% to 85.7% and achieve a 1.71x reduction in time cost compared to SOTA GreenFuzz while maintaining comparable recall. We further observe that combining our approach with existing unit test generation methods yields complementary gains, increasing the bug detection recall from 67.3% to 79.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18868v2">KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 19 pages (9 pages main text), 4 figures. v2: Full revision
    </div>
    <details class="paper-abstract">
      High-performance GPU kernels are critical for efficient LLM serving, yet their optimization remains a bottleneck requiring deep system expertise. While code LLMs show promise in generating functionally correct code, kernel optimization is intrinsically a search problem over a vast optimization space. The fundamental mismatch prevents existing LLM agents from efficiently exploring the optimization space for diverse hardware and compute patterns. To bridge the gap, we present KernelBand, a framework that formulates kernel optimization as a Multi-Armed Bandit (MAB) problem, explicitly balancing exploration and exploitation to unlock the potential of code LLMs. To navigate the infinite arm space of optimization strategies applied to candidate kernels, we design two key mechanisms: a hardware-aware pruning strategy via profiling bounds and a trace-driven clustering algorithm that leverages Lipschitz continuity. Theoretically, we prove that KernelBand reduces the regret bound to depend on the compact covering number of runtime clusters, ensuring sample-efficient discovery of high-performance kernels. Extensive experiments on TritonBench-G with three GPU architectures and four code LLMs show that KernelBand consistently and substantially outperforms state-of-the-art methods with over 33% average improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10388v1">Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10387v1">Making Databases Faster with LLM Evolutionary Sampling</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Traditional query optimization relies on cost-based optimizers that estimate execution cost (e.g., runtime, memory, and I/O) using predefined heuristics and statistical models. Improving these heuristics requires substantial engineering effort, and even when implemented, these heuristics often cannot take into account semantic correlations in queries and schemas that could enable better physical plans. Using our DBPlanBench harness for the DataFusion engine, we expose the physical plan through a compact serialized representation and let the LLM propose localized edits that can be applied and executed. We then apply an evolutionary search over these edits to refine candidates across iterations. Our key insight is that LLMs can leverage semantic knowledge to identify and apply non-obvious optimizations, such as join orderings that minimize intermediate cardinalities. We obtain up to 4.78$\times$ speedups on some queries and we demonstrate a small-to-large workflow in which optimizations found on small databases transfer effectively to larger databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11444v1">Towards Reliable Machine Translation: Scaling LLMs for Critical Error Detection and Safety</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at ECIR 2026
    </div>
    <details class="paper-abstract">
      Machine Translation (MT) plays a pivotal role in cross-lingual information access, public policy communication, and equitable knowledge dissemination. However, critical meaning errors, such as factual distortions, intent reversals, or biased translations, can undermine the reliability, fairness, and safety of multilingual systems. In this work, we explore the capacity of instruction-tuned Large Language Models (LLMs) to detect such critical errors, evaluating models across a range of parameters using the publicly accessible data sets. Our findings show that model scaling and adaptation strategies (zero-shot, few-shot, fine-tuning) yield consistent improvements, outperforming encoder-only baselines like XLM-R and ModernBERT. We argue that improving critical error detection in MT contributes to safer, more trustworthy, and socially accountable information systems by reducing the risk of disinformation, miscommunication, and linguistic harm, especially in high-stakes or underrepresented contexts. This work positions error detection not merely as a technical challenge, but as a necessary safeguard in the pursuit of just and responsible multilingual AI. The code will be made available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11388v1">Sparse Semantic Dimension as a Generalization Certificate for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Work in progress (17 pages)
    </div>
    <details class="paper-abstract">
      Standard statistical learning theory predicts that Large Language Models (LLMs) should overfit because their parameter counts vastly exceed the number of training tokens. Yet, in practice, they generalize robustly. We propose that the effective capacity controlling generalization lies in the geometry of the model's internal representations: while the parameter space is high-dimensional, the activation states lie on a low-dimensional, sparse manifold. To formalize this, we introduce the Sparse Semantic Dimension (SSD), a complexity measure derived from the active feature vocabulary of a Sparse Autoencoder (SAE) trained on the model's layers. Treating the LLM and SAE as frozen oracles, we utilize this framework to attribute the model's generalization capabilities to the sparsity of the dictionary rather than the total parameter count. Empirically, we validate this framework on GPT-2 Small and Gemma-2B, demonstrating that our bound provides non-vacuous certificates at realistic sample sizes. Crucially, we uncover a counter-intuitive "feature sharpness" scaling law: despite being an order of magnitude larger, Gemma-2B requires significantly fewer calibration samples to identify its active manifold compared to GPT-2, suggesting that larger models learn more compressible, distinct semantic structures. Finally, we show that this framework functions as a reliable safety monitor: out-of-distribution inputs trigger a measurable "feature explosion" (a sharp spike in active features), effectively signaling epistemic uncertainty through learned feature violation. Code is available at: https://github.com/newcodevelop/sparse-semantic-dimension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19905v2">Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonically with increasing inference time samples $k$. However, the specific reward that optimizes inference-time selection generally differs from the teacher. In contrast, substantial reward misspecification induces a finite optimal $k$ beyond which more sampling can increase the generalization error. For fixed $k$, there exists an optimal sampling temperature. We experimentally verify these facts in large language model inference with an additional large language model as a judge. In the "best-of-$k$" limit with the teacher as reward, we theoretically show that the generalization error decays as $Θ(1/k^2)$ and determine the leading coefficient via extreme value theory. These formulas delineate domains where scaling inference-time computation is provably preferable to collecting more data. Finally, we demonstrate that when task difficulty increases, the previously mentioned advantage of inference-time compute degrades.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09316v2">Effective MoE-based LLM Compression by Exploiting Heterogeneous Inter-Group Experts Routing Frequency and Information Density</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) based Large Language Models (LLMs) have achieved superior performance, yet the massive memory overhead caused by storing multiple expert networks severely hinders their practical deployment. Singular Value Decomposition (SVD)-based compression has emerged as a promising post-training technique; however, most existing methods apply uniform rank allocation or rely solely on static weight properties. This overlooks the substantial heterogeneity in expert utilization observed in MoE models, where frequent routing patterns and intrinsic information density vary significantly across experts. In this work, we propose RFID-MoE, an effective framework for MoE compression by exploiting heterogeneous Routing Frequency and Information Density. We first introduce a fused metric that combines expert activation frequency with effective rank to measure expert importance, adaptively allocating higher ranks to critical expert groups under a fixed budget. Moreover, instead of discarding compression residuals, we reconstruct them via a parameter-efficient sparse projection mechanism to recover lost information with minimal parameter overhead. Extensive experiments on representative MoE LLMs (e.g., Qwen3, DeepSeekMoE) across multiple compression ratios demonstrate that RFID-MoE consistently outperforms state-of-the-art methods like MoBE and D2-MoE. Notably, RFID-MoE achieves a perplexity of 16.92 on PTB with the Qwen3-30B model at a 60% compression ratio, reducing perplexity by over 8.0 compared to baselines, and improves zero-shot accuracy on HellaSwag by approximately 8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04860v2">Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark both open and closed-source LLMs. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide limited defenses against alignment tipping. These findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at https://github.com/aiming-lab/ATP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11361v1">Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive performance across a variety of reasoning tasks. However, their problem-solving ability often declines on more complex tasks due to hallucinations and the accumulation of errors within these intermediate steps. Recent work has introduced the notion of critical tokens--tokens in the reasoning process that exert significant influence on subsequent steps. Prior studies suggest that replacing critical tokens can refine reasoning trajectories. Nonetheless, reliably identifying and exploiting critical tokens remains challenging. To address this, we propose the Paraphrastic Probing and Consistency Verification~(PPCV) framework. PPCV operates in two stages. In the first stage, we roll out an initial reasoning path from the original question and then concatenate paraphrased versions of the question with this reasoning path. And we identify critical tokens based on mismatches between the predicted top-1 token and the expected token in the reasoning path. A criterion is employed to confirm the final critical token. In the second stage, we substitute critical tokens with candidate alternatives and roll out new reasoning paths for both the original and paraphrased questions. The final answer is determined by checking the consistency of outputs across these parallel reasoning processes. We evaluate PPCV on mainstream LLMs across multiple benchmarks. Extensive experiments demonstrate PPCV substantially enhances the reasoning performance of LLMs compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11354v1">ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at https://github.com/CenterForOpenScience/llm-benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00513v2">Minerva: Reinforcement Learning with Verifiable Rewards for Cyber Threat Intelligence LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Cyber threat intelligence (CTI) analysts routinely convert noisy, unstructured security artifacts into standardized, automation-ready representations. Although large language models (LLMs) show promise for this task, existing approaches remain brittle when producing structured CTI outputs and have largely relied on supervised fine-tuning (SFT). In contrast, CTI standards and community-maintained resources define canonical identifiers and schemas that enable deterministic verification of model outputs. We leverage this structure to study reinforcement learning with verifiable rewards (RLVR) for CTI tasks. We introduce \textit{Minerva}, a unified dataset and training pipeline spanning multiple CTI subtasks, each paired with task-specific verifiers that score structured outputs and identifier predictions. To address reward sparsity during rollout, we propose a lightweight self-training mechanism that generates additional verified trajectories and distills them back into the model. Experiments across LLM backbones show consistent improvements in accuracy and robustness over SFT across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11348v1">AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11340v1">Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become widely adopted as automated judges for evaluating AI-generated content. Despite their success, aligning LLM-based evaluations with human judgments remains challenging. While supervised fine-tuning on human-labeled data can improve alignment, it is costly and inflexible, requiring new training for each task or dataset. Recent progress in auto prompt optimization (APO) offers a more efficient alternative by automatically improving the instructions that guide LLM judges. However, existing APO methods primarily target text-only evaluations and remain underexplored in multimodal settings. In this work, we study auto prompt optimization for multimodal LLM-as-a-judge, particularly for evaluating AI-generated images. We identify a key bottleneck: multimodal models can only process a limited number of visual examples due to context window constraints, which hinders effective trial-and-error prompt refinement. To overcome this, we propose BLPO, a bi-level prompt optimization framework that converts images into textual representations while preserving evaluation-relevant visual cues. Our bi-level optimization approach jointly refines the judge prompt and the I2T prompt to maintain fidelity under limited context budgets. Experiments on four datasets and three LLM judges demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13811v5">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Published as a conference paper at COLM 2025 (The new version has been polished and expanded with more detailed future work ideas)
    </div>
    <details class="paper-abstract">
      WebShell attacks - where adversaries implant malicious scripts on web servers - remain a persistent threat. Prior machine-learning and deep-learning detectors typically depend on task-specific supervision and can be brittle under data scarcity, rapid concept drift, and out-of-distribution (OOD) deployment. Large language models (LLMs) have recently shown strong code understanding capabilities, but their reliability for WebShell detection remains unclear. We address this gap by (i) systematically evaluating seven LLMs (including GPT-4, LLaMA-3.1-70B, and Qwen-2.5 variants) against representative sequence- and graph-based baselines on 26.59K PHP scripts, and (ii) proposing Behavioral Function-Aware Detection (BFAD), a behavior-centric framework that adapts LLM inference to WebShell-specific execution patterns. BFAD anchors analysis on security-sensitive PHP functions via a Critical Function Filter, constructs compact LLM inputs with Context-Aware Code Extraction, and selects in-context demonstrations using Weighted Behavioral Function Profiling, which ranks examples by a behavior-weighted, function-level similarity. Empirically, we observe a consistent precision-recall asymmetry: larger LLMs often achieve high precision but miss attacks (lower recall), while smaller models exhibit the opposite tendency; moreover, off-the-shelf LLM prompting underperforms established detectors. BFAD substantially improves all evaluated LLMs, boosting F1 by 13.82% on average; notably, GPT-4, LLaMA-3.1-70B, and Qwen-2.5-Coder-14B exceed prior SOTA benchmarks, while Qwen-2.5-Coder-3B becomes competitive with traditional methods. Overall, our results clarify when LLMs succeed or fail on WebShell detection, provide a practical recipe, and highlight future directions for making LLM-based detection more reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10117v2">Biases in the Blind Spot: Detecting What LLMs Fail to Mention</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often provide chain-of-thought (CoT) reasoning traces that appear plausible, but may hide internal biases. We call these *unverbalized biases*. Monitoring models via their stated reasoning is therefore unreliable, and existing bias evaluations typically require predefined categories and hand-crafted datasets. In this work, we introduce a fully automated, black-box pipeline for detecting task-specific unverbalized biases. Given a task dataset, the pipeline uses LLM autoraters to generate candidate bias concepts. It then tests each concept on progressively larger input samples by generating positive and negative variations, and applies statistical techniques for multiple testing and early stopping. A concept is flagged as an unverbalized bias if it yields statistically significant performance differences while not being cited as justification in the model's CoTs. We evaluate our pipeline across six LLMs on three decision tasks (hiring, loan approval, and university admissions). Our technique automatically discovers previously unknown biases in these models (e.g., Spanish fluency, English proficiency, writing formality). In the same run, the pipeline also validates biases that were manually identified by prior work (gender, race, religion, ethnicity). More broadly, our proposed approach provides a practical, scalable path to automatic task-specific bias discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11328v1">Evaluating Alignment of Behavioral Dispositions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      As LLMs integrate into our daily lives, understanding their behavior becomes essential. In this work, we focus on behavioral dispositions$-$the underlying tendencies that shape responses in social contexts$-$and introduce a framework to study how closely the dispositions expressed by LLMs align with those of humans. Our approach is grounded in established psychological questionnaires but adapts them for LLMs by transforming human self-report statements into Situational Judgment Tests (SJTs). These SJTs assess behavior by eliciting natural recommendations in realistic user-assistant scenarios. We generate 2,500 SJTs, each validated by three human annotators, and collect preferred actions from 10 annotators per SJT, from a large pool of 550 participants. In a comprehensive study involving 25 LLMs, we find that models often do not reflect the distribution of human preferences: (1) in scenarios with low human consensus, LLMs consistently exhibit overconfidence in a single response; (2) when human consensus is high, smaller models deviate significantly, and even some frontier models do not reflect the consensus in 15-20% of cases; (3) traits can exhibit cross-LLM patterns, e.g., LLMs may encourage emotion expression in contexts where human consensus favors composure. Lastly, mapping psychometric statements directly to behavioral scenarios presents a unique opportunity to evaluate the predictive validity of self-reports, revealing considerable gaps between LLMs' stated values and their revealed behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11304v1">CryptoAnalystBench: Failures in Multi-Tool Long-Form LLM Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Modern analyst agents must reason over complex, high token inputs, including dozens of retrieved documents, tool outputs, and time sensitive data. While prior work has produced tool calling benchmarks and examined factuality in knowledge augmented systems, relatively little work studies their intersection: settings where LLMs must integrate large volumes of dynamic, structured and unstructured multi tool outputs. We investigate LLM failure modes in this regime using crypto as a representative high data density domain. We introduce (1) CryptoAnalystBench, an analyst aligned benchmark of 198 production crypto and DeFi queries spanning 11 categories; (2) an agentic harness equipped with relevant crypto and DeFi tools to generate responses across multiple frontier LLMs; and (3) an evaluation pipeline with citation verification and an LLM as a judge rubric spanning four user defined success dimensions: relevance, temporal relevance, depth, and data consistency. Using human annotation, we develop a taxonomy of seven higher order error types that are not reliably captured by factuality checks or LLM based quality scoring. We find that these failures persist even in state of the art systems and can compromise high stakes decisions. Based on this taxonomy, we refine the judge rubric to better capture these errors. While the judge does not align with human annotators on precise scoring across rubric iterations, it reliably identifies critical failure modes, enabling scalable feedback for developers and researchers studying analyst style agents. We release CryptoAnalystBench with annotated queries, the evaluation pipeline, judge rubrics, and the error taxonomy, and outline mitigation strategies and open challenges in evaluating long form, multi tool augmented systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19287v2">PALM: Path-aware LLM-based Test Generation with Comprehension</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 11 pages, v2
    </div>
    <details class="paper-abstract">
      Symbolic execution is a widely used technique for test generation, offering systematic exploration of program paths through constraint solving. However, it is fundamentally constrained by the capability to model the target code, including library functions, in terms of symbolic constraints and by the capability of underlying constraint solvers. As a result, many paths involving complex features remain unanalyzed or insufficiently modeled. Recent advances in large language models (LLMs) have shown promise in generating diverse and valid test inputs. Yet, LLMs lack mechanisms for systematically enumerating program paths and often fail to cover subtle corner cases. We observe that directly prompting an LLM with the full program leads to missed coverage of interesting paths. In this paper, we present PALM, a test generation system that combines symbolic path enumeration with LLM-assisted test generation. PALM statically enumerates possible paths through AST-level analysis and transforms each into an executable variant with embedded assertions that specify the target path. This avoids the need to translate path constraints into SMT formulas, by instead constructing program variants that the LLM can interpret. Importantly, PALM provides an interactive frontend that visualizes path coverage alongside generated tests, assembling tests based on the specific paths they exercise. A user study with 12 participants demonstrates that PALM's frontend helps users better understand path coverage and identify which paths are actually exercised by PALM-generated tests through verification and visualization of their path profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11089v1">DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      In the current landscape of Large Language Models (LLMs), the curation of large-scale, high-quality training data is a primary driver of model performance. A key lever is the \emph{data recipe}, which comprises a data processing pipeline to transform raw sources into training corpora. Despite the growing use of LLMs to automate individual data processing steps, such as data synthesis and filtering, the overall design of data recipes remains largely manual and labor-intensive, requiring substantial human expertise and iteration. To bridge this gap, we formulate \emph{end-to-end data recipe generation} for LLM adaptation. Given a target benchmark and a pool of available data sources, a model is required to output a complete data recipe that adapts a base LLM to the target task. We present DataChef-32B, which performs online reinforcement learning using a proxy reward that predicts downstream performance for candidate recipes. Across six held-out tasks, DataChef-32B produces practical recipes that reach comparable downstream performance to those curated by human experts. Notably, the recipe from DataChef-32B adapts Qwen3-1.7B-Base to the math domain, achieving 66.7 on AIME'25 and surpassing Qwen3-1.7B. This work sheds new light on automating LLM training and developing self-evolving AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11088v1">Vulnerabilities in Partial TEE-Shielded LLM Inference with Precomputed Noise</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) on third-party devices requires new ways to protect model intellectual property. While Trusted Execution Environments (TEEs) offer a promising solution, their performance limits can lead to a critical compromise: using a precomputed, static secret basis to accelerate cryptographic operations. We demonstrate that this mainstream design pattern introduces a classic cryptographic flaw, the reuse of secret keying material, into the system's protocol. We prove its vulnerability with two distinct attacks: First, our attack on a model confidentiality system achieves a full confidentiality break by recovering its secret permutations and model weights. Second, our integrity attack completely bypasses the integrity checks of systems like Soter and TSQP. We demonstrate the practicality of our attacks against state-of-the-art LLMs, recovering a layer's secrets from a LLaMA-3 8B model in about 6 minutes and showing the attack scales to compromise 405B-parameter LLMs across a variety of configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11247v1">Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Multi-turn prompt injection attacks distribute malicious intent across multiple conversation turns, exploiting the assumption that each turn is evaluated independently. While single-turn detection has been extensively studied, no published formula exists for aggregating per-turn pattern scores into a conversation-level risk score at the proxy layer -- without invoking an LLM. We identify a fundamental flaw in the intuitive weighted-average approach: it converges to the per-turn score regardless of turn count, meaning a 20-turn persistent attack scores identically to a single suspicious turn. Drawing on analogies from change-point detection (CUSUM), Bayesian belief updating, and security risk-based alerting, we propose peak + accumulation scoring -- a formula combining peak single-turn risk, persistence ratio, and category diversity. Evaluated on 10,654 multi-turn conversations -- 588 attacks sourced from WildJailbreak adversarial prompts and 10,066 benign conversations from WildChat -- the formula achieves 90.8% recall at 1.20% false positive rate with an F1 of 85.9%. A sensitivity analysis over the persistence parameter reveals a phase transition at rho ~ 0.4, where recall jumps 12 percentage points with negligible FPR increase. We release the scoring algorithm, pattern library, and evaluation harness as open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11083v1">Token-Efficient Change Detection in LLM APIs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Remote change detection in LLMs is a difficult problem. Existing methods are either too expensive for deployment at scale, or require initial white-box access to model weights or grey-box access to log probabilities. We aim to achieve both low cost and strict black-box operation, observing only output tokens. Our approach hinges on specific inputs we call Border Inputs, for which there exists more than one output top token. From a statistical perspective, optimal change detection depends on the model's Jacobian and the Fisher information of the output distribution. Analyzing these quantities in low-temperature regimes shows that border inputs enable powerful change detection tests. Building on this insight, we propose the Black-Box Border Input Tracking (B3IT) scheme. Extensive in-vivo and in-vitro experiments show that border inputs are easily found for non-reasoning tested endpoints, and achieve performance on par with the best available grey-box approaches. B3IT reduces costs by $30\times$ compared to existing methods, while operating in a strict black-box setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11079v1">In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      We propose activation-based data attribution, a method that traces behavioral changes in post-trained language models to responsible training datapoints. By computing activation-difference vectors for both test prompts and preference pairs and ranking by cosine similarity, we identify datapoints that cause specific behaviors and validate these attributions causally by retraining with modified data. Clustering behavior-datapoint similarity matrices also enables unsupervised discovery of emergent behaviors. Applying this to OLMo 2's production DPO training, we surfaced distractor-triggered compliance: a harmful behavior where the model complies with dangerous requests when benign formatting instructions are appended. Filtering top-ranked datapoints reduces this behavior by 63% while switching their labels achieves 78%. Our method outperforms gradient-based attribution and LLM-judge baselines while being over 10 times cheaper than both. This in-the-wild model organism - emerging from contaminated preference data rather than deliberate injection - provides a realistic benchmark for safety techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11243v1">Evaluating Memory Structure in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Modern LLM-based agents and chat assistants rely on long-term memory frameworks to store reusable knowledge, recall user preferences, and augment reasoning. As researchers create more complex memory architectures, it becomes increasingly difficult to analyze their capabilities and guide future memory designs. Most long-term memory benchmarks focus on simple fact retention, multi-hop recall, and time-based changes. While undoubtedly important, these capabilities can often be achieved with simple retrieval-augmented LLMs and do not test complex memory hierarchies. To bridge this gap, we propose StructMemEval - a benchmark that tests the agent's ability to organize its long-term memory, not just factual recall. We gather a suite of tasks that humans solve by organizing their knowledge in a specific structure: transaction ledgers, to-do lists, trees and others. Our initial experiments show that simple retrieval-augmented LLMs struggle with these tasks, whereas memory agents can reliably solve them if prompted how to organize their memory. However, we also find that modern LLMs do not always recognize the memory structure when not prompted to do so. This highlights an important direction for future improvements in both LLM training and memory frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11052v1">GraphSeek: Next-Generation Graph Analytics with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Graphs are foundational across domains but remain hard to use without deep expertise. LLMs promise accessible natural language (NL) graph analytics, yet they fail to process industry-scale property graphs effectively and efficiently: such datasets are large, highly heterogeneous, structurally complex, and evolve dynamically. To address this, we devise a novel abstraction for complex multi-query analytics over such graphs. Its key idea is to replace brittle generation of graph queries directly from NL with planning over a Semantic Catalog that describes both the graph schema and the graph operations. Concretely, this induces a clean separation between a Semantic Plane for LLM planning and broader reasoning, and an Execution Plane for deterministic, database-grade query execution over the full dataset and tool implementations. This design yields substantial gains in both token efficiency and task effectiveness even with small-context LLMs. We use this abstraction as the basis of the first LLM-enhanced graph analytics framework called GraphSeek. GraphSeek achieves substantially higher success rates (e.g., 86% over enhanced LangChain) and points toward the next generation of affordable and accessible graph analytics that unify LLM reasoning with database-grade execution over large and complex property graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24303v3">Retrieval- and Argumentation-Enhanced Multi-Agent LLMs for Judgmental Forecasting (Extended Version with Supplementary Material)</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 24 pages, 3 figures, Accepted to AAMAS 2026
    </div>
    <details class="paper-abstract">
      Judgmental forecasting is the task of making predictions about future events based on human judgment. This task can be seen as a form of claim verification, where the claim corresponds to a future event and the task is to assess the plausibility of that event. In this paper, we propose a novel multi-agent framework for claim verification, whereby different agents may disagree on claim veracity and bring specific evidence for and against the claims, represented as quantitative bipolar argumentation frameworks (QBAFs). We then instantiate the framework for supporting claim verification, with a variety of agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an existing approach for claim verification that generates and evaluates QBAFs; (2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM) from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents, extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of arguments from external sources. Finally, we conduct experiments with two standard judgmental forecasting datasets, with instances of our framework with two or three agents, empowered by six different base LLMs. We observe that combining evidence from agents can improve forecasting accuracy, especially in the case of three agents, while providing an explainable combination of evidence for claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11924v3">Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction refers to the phenomenon where a language model refines its own outputs purely through prompting, without external feedback or parameter updates. While this approach improves performance across diverse tasks, its mechanism remains unclear. We show that intrinsic self-correction functions by steering hidden representations along interpretable latent directions, as evidenced by both alignment analysis and activation interventions. To achieve this, we analyze intrinsic self-correction via the representation shift induced by prompting. In parallel, we construct interpretable latent directions with contrastive pairs and verify the causal effect of these directions via activation addition. Evaluating six open-source LLMs, our results demonstrate that prompt-induced representation shifts in text detoxification and text toxification consistently align with latent directions constructed from contrastive pairs. In detoxification, the shifts align with the non-toxic direction; in toxification, they align with the toxic direction. These findings suggest that representation steering is the mechanistic driver of intrinsic self-correction. Our analysis highlights that understanding model internals offers a direct route to analyzing the mechanisms of prompt-driven LLM behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04900v3">Evaluating Kubernetes Performance for GenAI Inference: From Automatic Speech Recognition to LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 A accepted at the 17th International Conference on Performance Engineering
    </div>
    <details class="paper-abstract">
      As Generative AI (GenAI), particularly inference, rapidly emerges as a dominant workload category, the Kubernetes ecosystem is proactively evolving to natively support its unique demands. This industry paper demonstrates how emerging Kubernetes-native projects can be combined to deliver the benefits of container orchestration, such as scalability and resource efficiency, to complex AI workflows. We implement and evaluate an illustrative, multi-stage use case consisting of automatic speech recognition and summarization. First, we address batch inference by using Kueue to manage jobs that transcribe audio files with Whisper models and Dynamic Accelerator Slicer (DAS) to increase parallel job execution. Second, we address a discrete online inference scenario by feeding the transcripts to a Large Language Model for summarization hosted using llm-d, a novel solution utilizing the recent developments around the Kubernetes Gateway API Inference Extension (GAIE) for optimized routing of inference requests. Our findings illustrate that these complementary components (Kueue, DAS, and GAIE) form a cohesive, high-performance platform, proving Kubernetes' capability to serve as a unified foundation for demanding GenAI workloads: Kueue reduced total makespan by up to 15%; DAS shortened mean job completion time by 36\%; and GAIE working in conjunction with llm-d improved tail Time to First Token latency by up to 90% even under high loads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13553v2">LLM-Mediated Guidance of MARL Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      In complex multi-agent environments, achieving efficient learning and desirable behaviours is a significant challenge for Multi-Agent Reinforcement Learning (MARL) systems. This work explores the potential of combining MARL with Large Language Model (LLM)-mediated interventions to guide agents toward more desirable behaviours. Specifically, we investigate how LLMs can be used to interpret and facilitate interventions that shape the learning trajectories of multiple agents. We experimented with two types of interventions, referred to as controllers: a Natural Language (NL) Controller and a Rule-Based (RB) Controller. The RB Controller showed a stronger impact than the NL Controller, which uses a small (7B/8B) LLM to simulate human-like interventions. Our findings indicate that agents particularly benefit from early interventions, leading to more efficient training and higher performance. Both intervention types outperform the baseline without interventions, highlighting the potential of LLM-mediated guidance to accelerate training and enhance MARL performance in challenging environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10986v1">TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Abhishek Vijaya Kumar and Bhaskar Kataria have equal contribution
    </div>
    <details class="paper-abstract">
      In RL post-training of LLM agents, calls to external tools take several seconds or even minutes, leaving allocated GPUs idle and inflating post-training time and cost. While many tool invocations repeat across parallel rollouts and could in principle be cached, naively caching their outputs for reuse is incorrect since tool outputs depend on the environment state induced by prior agent interactions. We present TVCACHE, a stateful tool-value cache for LLM agent post-training. TVCACHE maintains a tree of observed tool-call sequences and performs longest-prefix matching for cache lookups: a hit occurs only when the agent's full tool history matches a previously executed sequence, guaranteeing identical environment state. On three diverse workloads-terminal-based tasks, SQL generation, and video understanding. TVCACHE achieves cache hit rates of up to 70% and reduces median tool call execution time by up to 6.9X, with no degradation in post-training reward accumulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10965v1">MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Knowledge editing (KE) enables precise modifications to factual content in large language models (LLMs). Existing KE methods are largely designed for dense architectures, limiting their applicability to the increasingly prevalent sparse Mixture-of-Experts (MoE) models that underpin modern scalable LLMs. Although MoEs offer strong efficiency and capacity scaling, naively adapting dense-model editors is both computationally costly and prone to routing distribution shifts that undermine stability and consistency. To address these challenges, we introduce MoEEdit, the first routing-stable framework for parameter-modifying knowledge editing in MoE LLMs. Our method reparameterizes expert updates via per-expert null-space projections that keep router inputs invariant and thereby suppress routing shifts. The resulting block-structured optimization is solved efficiently with a block coordinate descent (BCD) solver. Experiments show that MoEEdit attains state-of-the-art efficacy and generalization while preserving high specificity and routing stability, with superior compute and memory efficiency. These results establish a robust foundation for scalable, precise knowledge editing in sparse LLMs and underscore the importance of routing-stable interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10964v1">Can LLMs Cook Jamaican Couscous? A Study of Cultural Novelty in Recipe Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 14 pages, 12 figures, conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate and shape cultural content, ranging from narrative writing to artistic production. While these models demonstrate impressive fluency and generative capacity, prior work has shown that they also exhibit systematic cultural biases, raising concerns about stereotyping, homogenization, and the erasure of culturally specific forms of expression. Understanding whether LLMs can meaningfully align with diverse cultures beyond the dominant ones remains a critical challenge. In this paper, we study cultural adaptation in LLMs through the lens of cooking recipes, a domain in which culture, tradition, and creativity are tightly intertwined. We build on the \textit{GlobalFusion} dataset, which pairs human recipes from different countries according to established measures of cultural distance. Using the same country pairs, we generate culturally adapted recipes with multiple LLMs, enabling a direct comparison between human and LLM behavior in cross-cultural content creation. Our analysis shows that LLMs fail to produce culturally representative adaptations. Unlike humans, the divergence of their generated recipes does not correlate with cultural distance. We further provide explanations for this gap. We show that cultural information is weakly preserved in internal model representations, that models inflate novelty in their production by misunderstanding notions such as creativity and tradition, and that they fail to identify adaptation with its associated countries and to ground it in culturally salient elements such as ingredients. These findings highlight fundamental limitations of current LLMs for culturally oriented generation and have important implications for their use in culturally sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11230v1">DiSCoKit: An Open-Source Toolkit for Deploying Live LLM Experiences in Survey Research</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Advancing social-scientific research of human-AI interaction dynamics and outcomes often requires researchers to deliver experiences with live large-language models (LLMs) to participants through online survey platforms. However, technical and practical challenges (from logging chat data to manipulating AI behaviors for experimental designs) often inhibit survey-based deployment of AI stimuli. We developed DiSCoKit--an open-source toolkit for deploying live LLM experiences (e.g., ones based on models delivered through Microsoft Azure portal) through JavaScript-enabled survey platforms (e.g., Qualtrics). This paper introduces that toolkit, explaining its scientific impetus, describes its architecture and operation, as well as its deployment possibilities and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10377v1">Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Vision-Language-Action Models (VLAs) have emerged as a key paradigm of Physical AI and are increasingly deployed in autonomous vehicles, robots, and smart spaces. In these resource-constrained on-device settings, selecting an appropriate large language model (LLM) backbone is a critical challenge: models must balance accuracy with strict inference latency and hardware efficiency constraints. This makes hardware-software co-design a game-changing requirement for on-device LLM deployment, where each hardware platform demands a tailored architectural solution. We propose a hardware co-design law that jointly captures model accuracy and inference performance. Specifically, we model training loss as an explicit function of architectural hyperparameters and characterise inference latency via roofline modelling. We empirically evaluate 1,942 candidate architectures on NVIDIA Jetson Orin, training 170 selected models for 10B tokens each to fit a scaling law relating architecture to training loss. By coupling this scaling law with latency modelling, we establish a direct accuracy-latency correspondence and identify the Pareto frontier for hardware co-designed LLMs. We further formulate architecture search as a joint optimisation over precision and performance, deriving feasible design regions under industrial hardware and application budgets. Our approach reduces architecture selection from months to days. At the same latency as Qwen2.5-0.5B on the target hardware, our co-designed architecture achieves 19.42% lower perplexity on WikiText-2. To our knowledge, this is the first principled and operational framework for hardware co-design scaling laws in on-device LLM deployment. We will make the code and related checkpoints publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10371v1">Simple LLM Baselines are Competitive for Model Diffing</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Standard LLM evaluations only test capabilities or dispositions that evaluators designed them for, missing unexpected differences such as behavioral shifts between model revisions or emergent misaligned tendencies. Model diffing addresses this limitation by automatically surfacing systematic behavioral differences. Recent approaches include LLM-based methods that generate natural language descriptions and sparse autoencoder (SAE)-based methods that identify interpretable features. However, no systematic comparison of these approaches exists nor are there established evaluation criteria. We address this gap by proposing evaluation metrics for key desiderata (generalization, interestingness, and abstraction level) and use these to compare existing methods. Our results show that an improved LLM-based baseline performs comparably to the SAE-based method while typically surfacing more abstract behavioral differences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10367v1">LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in high-stakes clinical settings demands rigorous and reliable evaluation. However, existing medical benchmarks remain static, suffering from two critical limitations: (1) data contamination, where test sets inadvertently leak into training corpora, leading to inflated performance estimates; and (2) temporal misalignment, failing to capture the rapid evolution of medical knowledge. Furthermore, current evaluation metrics for open-ended clinical reasoning often rely on either shallow lexical overlap (e.g., ROUGE) or subjective LLM-as-a-Judge scoring, both inadequate for verifying clinical correctness. To bridge these gaps, we introduce LiveMedBench, a continuously updated, contamination-free, and rubric-based benchmark that weekly harvests real-world clinical cases from online medical communities, ensuring strict temporal separation from model training data. We propose a Multi-Agent Clinical Curation Framework that filters raw data noise and validates clinical integrity against evidence-based medical principles. For evaluation, we develop an Automated Rubric-based Evaluation Framework that decomposes physician responses into granular, case-specific criteria, achieving substantially stronger alignment with expert physicians than LLM-as-a-Judge. To date, LiveMedBench comprises 2,756 real-world cases spanning 38 medical specialties and multiple languages, paired with 16,702 unique evaluation criteria. Extensive evaluation of 38 LLMs reveals that even the best-performing model achieves only 39.2%, and 84% of models exhibit performance degradation on post-cutoff cases, confirming pervasive data contamination risks. Error analysis further identifies contextual application-not factual knowledge-as the dominant bottleneck, with 35-48% of failures stemming from the inability to tailor medical knowledge to patient-specific constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v3">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Recent progress in LLMs has enabled understanding of audio signals, but has also exposed new safety risks arising from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition to enable effective black-box attacks. SACRED-Bench adopts three composition mechanisms: (a) overlap of harmful and benign speech, (b) mixture of benign speech with harmful non-speech audio, and (c) multi-speaker dialogue. These mechanisms focus on evaluating safety in settings where benign and harmful intents co-occur within a single auditory scene. Moreover, questions in SACRED-Bench are designed to implicitly refer to content in the audio, such that no explicit harmful information appears in the text prompt alone. Experiments demonstrate that even Gemini 2.5 Pro, a state-of-the-art proprietary LLM with safety guardrails fully enabled, still exhibits a 66% attack success rate. To bridge this gap, we propose SALMONN-Guard, the first guard model that jointly inspects speech, audio, and text for safety judgments, reducing the attack success rate to 20%. Our results highlight the need for audio-aware defenses to ensure the safety of multimodal LLMs. The dataset and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10354v1">Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Satellite foundation models produce dense embeddings whose physical interpretability remains poorly understood, limiting their integration into environmental decision systems. Using 12.1 million samples across the Continental United States (2017--2023), we first present a comprehensive interpretability analysis of Google AlphaEarth's 64-dimensional embeddings against 26 environmental variables spanning climate, vegetation, hydrology, temperature, and terrain. Combining linear, nonlinear, and attention-based methods, we show that individual embedding dimensions map onto specific land surface properties, while the full embedding space reconstructs most environmental variables with high fidelity (12 of 26 variables exceed $R^2 > 0.90$; temperature and elevation approach $R^2 = 0.97$). The strongest dimension-variable relationships converge across all three analytical methods and remain robust under spatial block cross-validation (mean $ΔR^2 = 0.017$) and temporally stable across all seven study years (mean inter-year correlation $r = 0.963$). Building on these validated interpretations, we then developed a Land Surface Intelligence system that implements retrieval-augmented generation over a FAISS-indexed embedding database of 12.1 million vectors, translating natural language environmental queries into satellite-grounded assessments. An LLM-as-Judge evaluation across 360 query--response cycles, using four LLMs in rotating generator, system, and judge roles, achieved weighted scores of $μ= 3.74 \pm 0.77$ (scale 1--5), with grounding ($μ= 3.93$) and coherence ($μ= 4.25$) as the strongest criteria. Our results demonstrate that satellite foundation model embeddings are physically structured representations that can be operationalized for environmental and geospatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24069v3">Can LLMs Reason Structurally? Benchmarking via the Lens of Data Structures</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are deployed on increasingly complex tasks that require multi-step decision-making. Understanding their algorithmic reasoning abilities is therefore crucial. However, we lack a diagnostic benchmark for evaluating this capability. We propose data structures as a principled lens: as fundamental building blocks of algorithms, they naturally probe structural reasoning-the ability to understand and manipulate relationships such as order, hierarchy, and connectivity that underpin algorithmic reasoning. We introduce DSR-Bench, spanning 20 data structures, 35 operations, and 4,140 problem instances. DSR-Bench features hierarchical task organization, fully automated generation and evaluation, and fine-grained diagnostics. Evaluating 13 state-of-the-art LLMs reveals critical limitations: the top-performing model achieves only 0.46/1 on challenging instances. Three auxiliary probes targeting more realistic usages expose further weaknesses: models perform poorly on spatial data and context-rich scenarios, and they struggle to reason over their own code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10324v1">Discovering Differences in Strategic Behavior Between Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in social and strategic scenarios, it becomes critical to understand where and why their behavior diverges from that of humans. While behavioral game theory (BGT) provides a framework for analyzing behavior, existing models do not fully capture the idiosyncratic behavior of humans or black-box, non-human agents like LLMs. We employ AlphaEvolve, a cutting-edge program discovery tool, to directly discover interpretable models of human and LLM behavior from data, thereby enabling open-ended discovery of structural factors driving human and LLM behavior. Our analysis on iterated rock-paper-scissors reveals that frontier LLMs can be capable of deeper strategic behavior than humans. These results provide a foundation for understanding structural differences driving differences in human and LLM behavior in strategic interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03667v2">Invisible Saboteurs: Sycophantic LLMs Mislead Novices in Problem-Solving Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 ACM CHI conference on Human Factors in Computing Systems (2026)
    </div>
    <details class="paper-abstract">
      Sycophancy, the tendency of LLM-based chatbots to express excessive agreement with their users, even when inappropriate, is emerging as a significant risk in human-AI interactions. However, the extent to which this affects human-LLM collaboration in complex problem-solving tasks is not well quantified, especially among novices who are prone to misconceptions. We created two LLM chatbots, one with high sycophancy and one with low sycophancy, and conducted a within-subjects experiment (n=24) in the context of debugging machine learning models to investigate the effect of sycophancy on users' mental models, workflows, reliance behaviors, and perceptions of the chatbots. Our findings show that users of the high sycophancy chatbot were less likely to correct their misconceptions and spent more time over-relying on unhelpful LLM responses, leading them to significantly worse performance in the task. Despite these impaired outcomes, a majority of users were unable to detect the presence of excessive sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10321v1">Single-Turn LLM Reformulation Powered Multi-Stage Hybrid Re-Ranking for Tip-of-the-Tongue Known-Item Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Retrieving known items from vague descriptions, Tip-of-the-Tongue (ToT) retrieval, remains a significant challenge. We propose using a single call to a generic 8B-parameter LLM for query reformulation, bridging the gap between ill-formed ToT queries and specific information needs. This method is particularly effective where standard Pseudo-Relevance Feedback fails due to poor initial recall. Crucially, our LLM is not fine-tuned for ToT or specific domains, demonstrating that gains stem from our prompting strategy rather than model specialization. Rewritten queries feed a multi-stage pipeline: sparse retrieval (BM25), dense/late-interaction reranking (Contriever, E5-large-v2, ColBERTv2), monoT5 cross-encoding, and list-wise reranking (Qwen 2.5 72B). Experiments on 2025 TREC-ToT datasets show that while raw queries yield poor performance, our lightweight pre-retrieval transformation improves Recall by 20.61%. Subsequent reranking improves nDCG@10 by 33.88%, MRR by 29.92%, and MAP@10 by 29.98%, offering a cost-effective intervention that unlocks the potential of downstream rankers. Code and data: https://github.com/debayan1405/TREC-TOT-2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10282v1">Linear-LLM-SCM: Benchmarking LLMs for Coefficient Elicitation in Linear-Gaussian Causal Models</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 16 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown potential in identifying qualitative causal relations, but their ability to perform quantitative causal reasoning -- estimating effect sizes that parametrize functional relationships -- remains underexplored in continuous domains. We introduce Linear-LLM-SCM, a plug-and-play benchmarking framework for evaluating LLMs on linear Gaussian structural causal model (SCM) parametrization when the DAG is given. The framework decomposes a DAG into local parent-child sets and prompts an LLM to produce a regression-style structural equation per node, which is aggregated and compared against available ground-truth parameters. Our experiments show several challenges in such benchmarking tasks, namely, strong stochasticity in the results in some of the models and susceptibility to DAG misspecification via spurious edges in the continuous domains. Across models, we observe substantial variability in coefficient estimates for some settings and sensitivity to structural and semantic perturbations, highlighting current limitations of LLMs as quantitative causal parameterizers. We also open-sourced the benchmarking framework so that researchers can utilize their DAGs and any off-the-shelf LLMs plug-and-play for evaluation in their domains effortlessly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14331v2">LLM Priors for ERM over Programs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We study program-learning methods that are efficient in both samples and computation. Classical learning theory suggests that when the target admits a short program description (for example, a short piece of ``Python code''), it can be learned from relatively few examples by performing ERM over the program class. However, this approach relies on enumerating candidate programs, which is typically exponential in the description length. In contrast, gradient-based training avoids explicit search, but for some families of short programs it can require exponentially many samples to succeed. We propose \textsc{LLM-PV}, a propose-and-verify recipe that enables ERM-style selection over a discrete program class without exhaustive enumeration. A pretrained LLM induces a proposal distribution over candidate programs; each proposal is executed, scored on a held-out validation set, and the best program is selected. The method uses no gradient updates and does not use validation feedback to adapt the sampling distribution. Across algorithmic tasks including parity variants, pattern matching, and primality testing, \textsc{LLM-PV} often recovers the exact underlying rule from a small labeled set and generalizes far beyond the training sequence lengths. In the same regimes, SGD-trained transformers and standard adaptation baselines (fine-tuning and in-context learning), as well as classical ML baselines, can fit the training data yet fail to generalize reliably. Together, these results suggest that pretrained LLM priors can serve as effective search biases for ERM, narrowing the gap between statistical and computational efficiency. The code is available at [\href{https://github.com/DLFundamentals/LLM_PV}{code}].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10246v1">KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Solid State Drives (SSDs) are critical to datacenters, consumer platforms, and mission-critical systems. Yet diagnosing their performance and reliability is difficult because data are fragmented and time-disjoint, and existing methods demand large datasets and expert input while offering only limited insights. Degradation arises not only from shifting workloads and evolving architectures but also from environmental factors such as temperature, humidity, and vibration. We present KORAL, a knowledge driven reasoning framework that integrates Large Language Models (LLMs) with a structured Knowledge Graph (KG) to generate insights into SSD operations. Unlike traditional approaches that require extensive expert input and large datasets, KORAL generates a Data KG from fragmented telemetry and integrates a Literature KG that already organizes knowledge from literature, reports, and traces. This turns unstructured sources into a queryable graph and telemetry into structured knowledge, and both the Graphs guide the LLM to deliver evidence-based, explainable analysis aligned with the domain vocabulary and constraints. Evaluation using real production traces shows that the KORAL delivers expert-level diagnosis and recommendations, supported by grounded explanations that improve reasoning transparency, guide operator decisions, reduce manual effort, and provide actionable insights to improve service quality. To our knowledge, this is the first end-to-end system that combines LLMs and KGs for full-spectrum SSD reasoning including Descriptive, Predictive, Prescriptive, and What-if analysis. We release the generated SSD-specific KG to advance reproducible research in knowledge-based storage system analysis. GitHub Repository: https://github.com/Damrl-lab/KORAL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10226v1">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Optimizing large-scale machine learning systems, such as recommendation models for global video platforms, requires navigating a massive hyperparameter search space and, more critically, designing sophisticated optimizers, architectures, and reward functions to capture nuanced user behaviors. Achieving substantial improvements in these areas is a non-trivial task, traditionally relying on extensive manual iterations to test new hypotheses. We propose a self-evolving system that leverages Large Language Models (LLMs), specifically those from Google's Gemini family, to autonomously generate, train, and deploy high-performing, complex model changes within an end-to-end automated workflow. The self-evolving system is comprised of an Offline Agent (Inner Loop) that performs high-throughput hypothesis generation using proxy metrics, and an Online Agent (Outer Loop) that validates candidates against delayed north star business metrics in live production. Our agents act as specialized Machine Learning Engineers (MLEs): they exhibit deep reasoning capabilities, discovering novel improvements in optimization algorithms and model architecture, and formulating innovative reward functions that target long-term user engagement. The effectiveness of this approach is demonstrated through several successful production launches at YouTube, confirming that autonomous, LLM-driven evolution can surpass traditional engineering workflows in both development velocity and model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10218v1">ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have sparked growing interest in applying them to hardware design automation, particularly for accurate RTL code generation. Prior efforts follow two largely independent paths: (i) training domain-adapted RTL models to internalize hardware semantics, (ii) developing agentic systems that leverage frontier generic LLMs guided by simulation feedback. However, these two paths exhibit complementary strengths and weaknesses. In this work, we present ACE-RTL that unifies both directions through Agentic Context Evolution (ACE). ACE-RTL integrates an RTL-specialized LLM, trained on a large-scale dataset of 1.7 million RTL samples, with a frontier reasoning LLM through three synergistic components: the generator, reflector, and coordinator. These components iteratively refine RTL code toward functional correctness. We further introduce a parallel scaling strategy that significantly reduces the number of iterations required to reach correct solutions. On the Comprehensive Verilog Design Problems (CVDP) benchmark, ACE-RTL achieves up to a 44.87% pass rate improvement over 14 competitive baselines while requiring only four iterations on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10210v1">How Much Reasoning Do Retrieval-Augmented Models Add beyond LLMs? A Benchmarking Framework for Multi-Hop Inference over Hybrid Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to struggle with knowledge-intensive questions that require up-to-date information and multi-hop reasoning. Augmenting LLMs with hybrid external knowledge, such as unstructured text and structured knowledge graphs, offers a promising alternative to costly continual pretraining. As such, reliable evaluation of their retrieval and reasoning capabilities becomes critical. However, many existing benchmarks increasingly overlap with LLM pretraining data, which means answers or supporting knowledge may already be encoded in model parameters, making it difficult to distinguish genuine retrieval and reasoning from parametric recall. We introduce HybridRAG-Bench, a framework for constructing benchmarks to evaluate retrieval-intensive, multi-hop reasoning over hybrid knowledge. HybridRAG-Bench automatically couples unstructured text and structured knowledge graph representations derived from recent scientific literature on arXiv, and generates knowledge-intensive question-answer pairs grounded in explicit reasoning paths. The framework supports flexible domain and time-frame selection, enabling contamination-aware and customizable evaluation as models and knowledge evolve. Experiments across three domains (artificial intelligence, governance and policy, and bioinformatics) demonstrate that HybridRAG-Bench rewards genuine retrieval and reasoning rather than parametric recall, offering a principled testbed for evaluating hybrid knowledge-augmented reasoning systems. We release our code and data at github.com/junhongmit/HybridRAG-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10092v1">Quantum-Audit: Evaluating the Reasoning Limits of LLMs on Quantum Computing</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Language models have become practical tools for quantum computing education and research, from summarizing technical papers to explaining theoretical concepts and answering questions about recent developments in the field. While existing benchmarks evaluate quantum code generation and circuit design, their understanding of quantum computing concepts has not been systematically measured. Quantum-Audit addresses this gap with 2,700 questions covering core quantum computing topics. We evaluate 26 models from leading organizations. Our benchmark comprises 1,000 expert-written questions, 1,000 questions extracted from research papers using LLMs and validated by experts, plus an additional 700 questions including 350 open-ended questions and 350 questions with false premises to test whether models can correct erroneous assumptions. Human participants scored between 23% and 86%, with experts averaging 74%. Top-performing models exceeded the expert average, with Claude Opus 4.5 reaching 84% accuracy, though top models showed an average 12-point accuracy drop on expert-written questions compared to LLM-generated ones. Performance declined further on advanced topics, dropping to 73% on security questions. Additionally, models frequently accepted and reinforced false premises embedded in questions instead of identifying them, with accuracy below 66% on these critical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08023v2">CyberExplorer: Benchmarking LLM Offensive Security Capabilities in a Real-World Attacking Simulation Environment</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Real-world offensive security operations are inherently open-ended: attackers explore unknown attack surfaces, revise hypotheses under uncertainty, and operate without guaranteed success. Existing LLM-based offensive agent evaluations rely on closed-world settings with predefined goals and binary success criteria. To address this gap, we introduce CyberExplorer, an evaluation suite with two core components: (1) an open-environment benchmark built on a virtual machine hosting 40 vulnerable web services derived from real-world CTF challenges, where agents autonomously perform reconnaissance, target selection, and exploitation without prior knowledge of vulnerability locations; and (2) a reactive multi-agent framework supporting dynamic exploration without predefined plans. CyberExplorer enables fine-grained evaluation beyond flag recovery, capturing interaction dynamics, coordination behavior, failure modes, and vulnerability discovery signals-bridging the gap between benchmarks and realistic multi-target attack scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03005v3">From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10017v1">SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support question answering and decision-making in high-stakes, domain-specific settings such as natural hazard response and infrastructure planning, where effective answers must convey fine-grained, decision-critical details. However, existing evaluation frameworks for retrieval-augmented generation (RAG) and open-ended question answering primarily rely on surface-level similarity, factual consistency, or semantic relevance, and often fail to assess whether responses provide the specific information required for domain-sensitive decisions. To address this gap, we propose a multi-dimensional, reference-free evaluation framework that assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. We introduce a curated dataset of 1,412 domain-specific question-answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation. We further conduct human evaluation to assess inter-annotator agreement and alignment between model outputs and human judgments, which highlights the inherent subjectivity of open-ended, domain-specific evaluation. Our results show that no single metric sufficiently captures answer quality in isolation and demonstrate the need for structured, multi-metric evaluation frameworks when deploying LLMs in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24968v3">The Impact of LLMs on Online News Consumption and Production</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) change how consumers acquire information online; their bots also crawl news publishers' websites for training data and to answer consumer queries; and they provide tools that can lower the cost of content creation. These changes lead to predictions of adverse impact on news publishers in the form of lowered consumer demand, reduced demand for newsroom employees, and an increase in news "slop." Consequently, some publishers strategically responded by blocking LLM access to their websites using the robots.txt file standard. Using high-frequency granular data, we document four effects related to the predicted shifts in news publishing following the introduction of generative AI (GenAI). First, we find a moderate decline in traffic to news publishers occurring after August 2024. Second, using a difference-in-differences approach, we find that blocking GenAI bots can be associated with a reduction of total website traffic to large publishers compared to not blocking. Third, on the hiring side, we do not find evidence that LLMs are replacing editorial or content-production jobs yet. The share of new editorial and content-production job listings increases over time. Fourth, regarding content production, we find no evidence that large publishers increased text volume; instead, they significantly increased rich content and use more advertising and targeting technologies. Together, these findings provide early evidence of some unforeseen impacts of the introduction of LLMs on news production and consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09944v1">Environment-in-the-Loop: Rethinking Code Migration with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Modern software systems continuously undergo code upgrades to enhance functionality, security, and performance, and Large Language Models (LLMs) have demonstrated remarkable capabilities in code migration tasks. However, while research on automated code migration which including refactoring, API adaptation, and dependency updates has advanced rapidly, the exploration of the automated environment interaction that must accompany it remains relatively scarce. In practice, code and its environment are intricately intertwined. Relying solely on static analysis of the environment leads to an inadequate understanding of the target setting, prolongs feedback cycles, and consequently causes significant rework and project delays, thereby reducing overall efficiency. We contend that successful software evolution demands a holistic perspective that integrates both code and environment migration. To understand the current landscape and challenges, we first provide an overview of the status of automated environment construction. We then propose a novel framework paradigm that tightly integrates automated environment setup with the code migration workflow. Finally, we explore the challenges and future directions for automated environment interaction within the code migration domain. Our findings emphasize that without automated environment interaction, the automation of code migration is only half complete.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07721v2">ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 25 pages, 16 figures. Under review
    </div>
    <details class="paper-abstract">
      KV-cache retrieval is essential for long-context LLM inference, yet existing methods struggle with distribution drift and high latency at scale. We introduce ParisKV, a drift-robust, GPU-native KV-cache retrieval framework based on collision-based candidate selection, followed by a quantized inner-product reranking estimator. For million-token contexts, ParisKV supports CPU-offloaded KV caches via Unified Virtual Addressing (UVA), enabling on-demand top-$k$ fetching with minimal overhead. ParisKV matches or outperforms full attention quality on long-input and long-generation benchmarks. It achieves state-of-the-art long-context decoding efficiency: it matches or exceeds full attention speed even at batch size 1 for long contexts, delivers up to 2.8$\times$ higher throughput within full attention's runnable range, and scales to million-token contexts where full attention runs out of memory. At million-token scale, ParisKV reduces decode latency by 17$\times$ and 44$\times$ compared to MagicPIG and PQCache, respectively, two state-of-the-art KV-cache Top-$k$ retrieval baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09930v1">JMigBench: A Benchmark for Evaluating LLMs on Source Code Migration (Java 8 to Java 11)</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We build a benchmark to evaluate large language models (LLMs) for source code migration tasks, specifically upgrading functions from Java 8 to Java 11. We first collected a dataset of function pairs from open-source repositories, but limitations in data quality led us to construct a refined dataset covering eight categories of deprecated APIs. Using this dataset, the Mistral Codestral model was evaluated with CodeBLEU and keyword-based metrics to measure lexical and semantic similarity as well as migration correctness. Results show that the evaluated model (Mistral Codestral) can handle trivial one-to-one API substitutions with moderate success, achieving identical migrations in 11.11% of the cases, but it struggles with more complex migrations such as CORBA or JAX-WS. These findings suggest Mistral Codestral can partially reduce developer effort by automating repetitive migration tasks but cannot yet replace humans within the scope of the JMigBench benchmark. The benchmark and analysis provide a foundation for future work on expanding datasets, refining prompting strategies, and improving migration performance across different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09924v1">LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: https://github.com/KabakaWilliam/llms_know_difficulty
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09902v1">Routing, Cascades, and User Choice for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 23 pages, accepted in ICLR 2026
    </div>
    <details class="paper-abstract">
      To mitigate the trade-offs between performance and costs, LLM providers route user tasks to different models based on task difficulty and latency. We study the effect of LLM routing with respect to user behavior. We propose a game between an LLM provider with two models (standard and reasoning) and a user who can re-prompt or abandon tasks if the routed model cannot solve them. The user's goal is to maximize their utility minus the delay from using the model, while the provider minimizes the cost of servicing the user. We solve this Stackelberg game by fully characterizing the user best response and simplifying the provider problem. We observe that in nearly all cases, the optimal routing policy involves a static policy with no cascading that depends on the expected utility of the models to the user. Furthermore, we reveal a misalignment gap between the provider-optimal and user-preferred routes when the user's and provider's rankings of the models with respect to utility and cost differ. Finally, we demonstrate conditions for extreme misalignment where providers are incentivized to throttle the latency of the models to minimize their costs, consequently depressing user utility. The results yield simple threshold rules for single-provider, single-user interactions and clarify when routing, cascading, and throttling help or harm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09901v1">QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Query Processing (QP) bridges user intent and content supply in large-scale Social Network Service (SNS) search engines. Traditional QP systems rely on pipelines of isolated discriminative models (e.g., BERT), suffering from limited semantic understanding and high maintenance overhead. While Large Language Models (LLMs) offer a potential solution, existing approaches often optimize sub-tasks in isolation, neglecting intrinsic semantic synergy and necessitating independent iterations. Moreover, standard generative methods often lack grounding in SNS scenarios, failing to bridge the gap between open-domain corpora and informal SNS linguistic patterns, while struggling to adhere to rigorous business definitions. We present QP-OneModel, a Unified Generative LLM for Multi-Task Query Understanding in the SNS domain. We reformulate heterogeneous sub-tasks into a unified sequence generation paradigm, adopting a progressive three-stage alignment strategy culminating in multi-reward Reinforcement Learning. Furthermore, QP-OneModel generates intent descriptions as a novel high-fidelity semantic signal, effectively augmenting downstream tasks such as query rewriting and ranking. Offline evaluations show QP-OneModel achieves a 7.35% overall gain over discriminative baselines, with significant F1 boosts in NER (+9.01%) and Term Weighting (+9.31%). It also exhibits superior generalization, surpassing a 32B model by 7.60% accuracy on unseen tasks. Fully deployed at Xiaohongshu, online A/B tests confirm its industrial value, optimizing retrieval relevance (DCG) by 0.21% and lifting user retention by 0.044%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12306v2">A large-scale pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English evaluative consider construction (consider X as/to be/zero Y). We annotate 143,933 'consider' concordance lines from the Corpus of Historical American English (COHA) via the OpenAI API in under 60 hours, achieving 98 percent+ accuracy on two sophisticated annotation procedures. A Bayesian multinomial GAM fitted to 44,527 true positives of the evaluative construction reveals previously undocumented genre-specific trajectories of change, enabling us to advance new hypotheses about the relationship between register formality and competing pressures of morphosyntactic reduction and enhancement. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, unlocking substantive research questions previously beyond practical reach, though implementation requires attention to costs, licensing, and other ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09851v1">CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian Hyperparameter Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Feature Engineering (FE) is pivotal in automated machine learning (AutoML) but remains a bottleneck for traditional methods, which treat it as a black-box search, operating within rigid, predefined search spaces and lacking domain awareness. While Large Language Models (LLMs) offer a promising alternative by leveraging semantic reasoning to generate unbounded operators, existing methods fail to construct free-form FE pipelines, remaining confined to isolated subtasks such as feature generation. Most importantly, they are rarely optimized jointly with hyperparameter optimization (HPO) of the ML model, leading to greedy "FE-then-HPO" workflows that cannot capture strong FE-HPO interactions. In this paper, we present CoFEH, a collaborative framework that interleaves LLM-based FE and Bayesian HPO for robust end-to-end AutoML. CoFEH uses an LLM-driven FE optimizer powered by Tree of Thought (ToT) to explore flexible FE pipelines, a Bayesian optimization (BO) module to solve HPO, and a dynamic optimizer selector that realizes interleaved optimization by adaptively scheduling FE and HPO steps. Crucially, we introduce a mutual conditioning mechanism that shares context between LLM and BO, enabling mutually informed decisions. Experiments show that CoFEH not only outperforms traditional and LLM-based FE baselines, but also achieves superior end-to-end performance under joint optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08493v2">LLM-based Vulnerable Code Augmentation: Generate or Refactor?</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 15 pages, Accepted by ESAAN 2026, version with added appendix
    </div>
    <details class="paper-abstract">
      Vulnerability code-bases often suffer from severe imbalance, limiting the effectiveness of Deep Learning-based vulnerability classifiers. Data Augmentation could help solve this by mitigating the scarcity of under-represented vulnerability types. In this context, we investigate LLM-based augmentation for vulnerable functions, comparing controlled generation of new vulnerable samples with semantics-preserving refactoring of existing ones. Using Qwen2.5-Coder to produce augmented data and CodeBERT as a classifier on the SVEN dataset, we find that our approaches are indeed effective in enriching vulnerable code-bases through a simple process and with reasonable quality, and that a hybrid strategy best boosts vulnerability classifiers' performance. Code repository is available here : https://github.com/DynaSoumhaneOuchebara/LLM-based-code-augmentation-Generate-or-Refactor-
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09832v1">LLM Reasoning Predicts When Models Are Right: Evidence from Coding Classroom Discourse</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed to automatically label and analyze educational dialogue at scale, yet current pipelines lack reliable ways to detect when models are wrong. We investigate whether reasoning generated by LLMs can be used to predict the correctness of a model's own predictions. We analyze 30,300 teacher utterances from classroom dialogue, each labeled by multiple state-of-the-art LLMs with an instructional move construct and an accompanying reasoning. Using human-verified ground-truth labels, we frame the task as predicting whether a model's assigned label for a given utterance is correct. We encode LLM reasoning using Term Frequency-Inverse Document Frequency (TF-IDF) and evaluate five supervised classifiers. A Random Forest classifier achieves an F1 score of 0.83 (Recall = 0.854), successfully identifying most incorrect predictions and outperforming baselines. Training specialist detectors for specific instructional move constructs further improves performance on difficult constructs, indicating that error detection benefits from construct-specific linguistic cues. Using the Linguistic Inquiry and Word Count (LIWC) framework, we examine four linguistic markers of correctness: Causation, Differentiation, Tentativeness, and Insight. Correct predictions exhibit grounded causal language (e.g., because, therefore), while incorrect reasoning is substantially more likely to rely on epistemic hedging (e.g., might, could) and performative metacognition (e.g., think, realize). Syntactic complexity does not distinguish correct from incorrect reasoning, and longer reasoning is not more reliable. These findings demonstrate that reasoning-based error detection offers a practical and scalable approach to quality control in automated educational dialogue analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09817v1">AnalyticsGPT: An LLM Workflow for Scientometric Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      This paper introduces AnalyticsGPT, an intuitive and efficient large language model (LLM)-powered workflow for scientometric question answering. This underrepresented downstream task addresses the subcategory of meta-scientific questions concerning the "science of science." When compared to traditional scientific question answering based on papers, the task poses unique challenges in the planning phase. Namely, the need for named-entity recognition of academic entities within questions and multi-faceted data retrieval involving scientometric indices, e.g. impact factors. Beyond their exceptional capacity for treating traditional natural language processing tasks, LLMs have shown great potential in more complex applications, such as task decomposition and planning and reasoning. In this paper, we explore the application of LLMs to scientometric question answering, and describe an end-to-end system implementing a sequential workflow with retrieval-augmented generation and agentic concepts. We also address the secondary task of effectively synthesizing the data into presentable and well-structured high-level analyses. As a database for retrieval-augmented generation, we leverage a proprietary research performance assessment platform. For evaluation, we consult experienced subject matter experts and leverage LLMs-as-judges. In doing so, we provide valuable insights on the efficacy of LLMs towards a niche downstream task. Our (skeleton) code and prompts are available at: https://github.com/lyvykhang/llm-agents-scientometric-qa/tree/acl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10171v1">EvoCodeBench: A Human-Performance Benchmark for Self-Evolving LLM-Driven Coding Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance in programming tasks, LLM-driven coding systems have evolved from one-shot code generation into complex systems capable of iterative improvement during inference. However, existing code benchmarks primarily emphasize static correctness and implicitly assume fixed model capability during inference. As a result, they do not capture inference-time self-evolution, such as whether accuracy and efficiency improve as an agent iteratively refines its solutions. They also provide limited accounting of resource costs and rarely calibrate model performance against that of human programmers. Moreover, many benchmarks are dominated by high-resource languages, leaving cross-language robustness and long-tail language stability underexplored. Therefore, we present EvoCodeBench, a benchmark for evaluating self-evolving LLM-driven coding systems across programming languages with direct comparison to human performance. EvoCodeBench tracks performance dynamics, measuring solution correctness alongside efficiency metrics such as solving time, memory consumption, and improvement algorithmic design over repeated problem-solving attempts. To ground evaluation in a human-centered reference frame, we directly compare model performance with that of human programmers on the same tasks, enabling relative performance assessment within the human ability distribution. Furthermore, EvoCodeBench supports multiple programming languages, enabling systematic cross-language and long-tail stability analyses under a unified protocol. Our results demonstrate that self-evolving systems exhibit measurable gains in efficiency over time, and that human-relative and multi-language analyses provide insights unavailable through accuracy alone. EvoCodeBench establishes a foundation for evaluating coding intelligence in evolving LLM-driven systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06133v3">LLM Serving Optimization with Variable Prefill and Decode Lengths</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We study offline scheduling for large language model (LLM) serving under a fixed KV-cache memory budget, where requests have heterogeneous prompt (prefill) and response (decode) lengths. Prompt tokens determine initial KV usage, and each generated token increases memory by one unit. Given a backlog of n requests arriving together, we schedule mixed prefill and decode batches to minimize total end-to-end latency. We show that heterogeneity in prompt lengths makes the problem computationally intractable and that widely used heuristics such as first-come-first-served and shortest-first can be arbitrarily suboptimal. We propose Sorted-F, which repeatedly forms feasible batches using a new selection metric that balances batch size against downstream decode cost, and prove it achieves a constant-factor guarantee on total latency. We further develop practical variants -- an exact solver for small instances and fast heuristics for larger ones -- and evaluate them on a public workload spanning short conversations and long-document summarization, where they consistently reduce average latency relative to standard baselines. Our results highlight that during peak-hour tidal backlogs, greedy GPU packing or short-request prioritization can perform poorly when prompt lengths vary widely, and provide a principled, tunable framework for designing production batch schedulers and planning capacity in memory-constrained LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11362v2">PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Understanding human behavior traits is central to applications in human-computer interaction, computational social science, and personalized AI systems. Such understanding often requires integrating multiple modalities to capture nuanced patterns and relationships. However, existing resources rarely provide datasets that combine behavioral descriptors with complementary modalities such as facial attributes and biographical information. To address this gap, we present PersonaX, a curated collection of multimodal datasets designed to enable comprehensive analysis of public traits across modalities. PersonaX consists of (1) CelebPersona, featuring 9444 public figures from diverse occupations, and (2) AthlePersona, covering 4181 professional athletes across 7 major sports leagues. Each dataset includes behavioral trait assessments inferred by three high-performing large language models, alongside facial imagery and structured biographical features. We analyze PersonaX at two complementary levels. First, we abstract high-level trait scores from text descriptions and apply five statistical independence tests to examine their relationships with other modalities. Second, we introduce a novel causal representation learning (CRL) framework tailored to multimodal and multi-measurement data, providing theoretical identifiability guarantees. Experiments on both synthetic and real-world data demonstrate the effectiveness of our approach. By unifying structured and unstructured analysis, PersonaX establishes a foundation for studying LLM-inferred behavioral traits in conjunction with visual and biographical attributes, advancing multimodal trait analysis and causal reasoning. The code is available at https://github.com/lokali/PersonaX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09719v1">Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Test-time adaptation (TTA) for large language models (LLMs) updates model parameters at inference time using signals available at deployment. This paper focuses on a common yet under-explored regime: unsupervised, sample-specific TTA, where the model adapts independently for each prompt using only the prompt itself, without gold answers or external supervision. Although appealing, naive unsupervised TTA with a fixed, handcrafted learning rate can be unstable: updates may overfit to prompt-specific statistics, drift from the desired answer distribution, and ultimately degrade generation quality. This failure mode is not surprising, as in this case TTA must adapt to a single prompt within only a few gradient steps, unlike standard training that averages updates over large datasets and long optimization horizons. Therefore, we propose layer-wise dynamic test-time adaptation, a framework which explicitly modulates TTA strength as a function of prompt representation, LLM structure and adaptation step. In our setting, TTA updates only LoRA parameters, and a lightweight hypernetwork predicts per-layer, per-step learning-rate multipliers, enabling fine-grained control. Experiments across various datasets and LLMs consistently show that our method substantially strengthens TTA by learning effective scaling patterns over adaptation steps and transformer layer projections, improving stability while delivering better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23631v2">Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted by The Fourteenth International Conference on Learning Representations (ICLR 2026)
    </div>
    <details class="paper-abstract">
      Alignment of large language models (LLMs) has predominantly relied on pairwise preference optimization, where annotators select the better of two responses to a prompt. While simple, this approach overlooks the opportunity to learn from richer forms of human feedback, such as multiway comparisons and top-$k$ rankings. We introduce Ranked Choice Preference Optimization (RCPO), a unified framework that bridges preference optimization with (ranked) choice modeling via maximum likelihood estimation. RCPO supports both utility-based and rank-based models, subsumes several pairwise methods (such as DPO and SimPO) as special cases, and provides principled training objectives for richer feedback formats. We instantiate this framework with two representative models (Multinomial Logit and Mallows-RMJ). Experiments on Llama-3-8B-Instruct, Gemma-2-9B-it, and Mistral-7B-Instruct across in-distribution and out-of-distribution settings show that RCPO consistently outperforms competitive baselines. RCPO shows that directly leveraging ranked preference data, combined with the right choice models, yields more effective alignment. It offers an extensible foundation for incorporating (ranked) choice modeling into LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09703v1">Maastricht University at AMIYA: Adapting LLMs for Dialectal Arabic using Fine-tuning and MBR Decoding</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly multilingual, supporting hundreds of languages, especially high resource ones. Unfortunately, Dialect variations are still underrepresented due to limited data and linguistic variation. In this work, we adapt a pre-trained LLM to improve dialectal performance. Specifically, we use Low Rank Adaptation (LoRA) fine-tuning on monolingual and English Dialect parallel data, adapter merging and dialect-aware MBR decoding to improve dialectal fidelity generation and translation. Experiments on Syrian, Moroccan, and Saudi Arabic show that merging and MBR improve dialectal fidelity while preserving semantic accuracy. This combination provides a compact and effective framework for robust dialectal Arabic generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10155v3">What Should Feature Distillation Transfer in LLMs? A Task-Tangent Geometry View</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Feature-based knowledge distillation aims to transfer intermediate representations from a teacher LLM model to a student. Existing approaches typically rely on direct feature matching or learned projections, implicitly treating representations as objects with intrinsic meaning. However, the relevance of a representation dimension is determined solely by how it affects the model's output. In this work, we propose a functional perspective on feature-based distillation. We characterize knowledge transfer in terms of the teacher's functional geometry, i.e., how its output depends on internal representations, rather than direct representation alignment. This viewpoint reveals that effective distillation need not preserve full high-dimensional features, but instead should retain dominant directions of functional contribution, naturally inducing an effective functional dimension for each task. Building on this framework, we introduce Flex-KD, an architecture-agnostic and parameter-free distillation method that transfers the teacher's functional geometry while matching the student's representational capacity. Extensive experiments across language understanding and generation benchmarks demonstrate that Flex-KD consistently outperforms existing distillation approaches, particularly under severe teacher-student dimension mismatch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09634v1">LLM-FS: Zero-Shot Feature Selection for Effective and Interpretable Malware Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Feature selection (FS) remains essential for building accurate and interpretable detection models, particularly in high-dimensional malware datasets. Conventional FS methods such as Extra Trees, Variance Threshold, Tree-based models, Chi-Squared tests, ANOVA, Random Selection, and Sequential Attention rely primarily on statistical heuristics or model-driven importance scores, often overlooking the semantic context of features. Motivated by recent progress in LLM-driven FS, we investigate whether large language models (LLMs) can guide feature selection in a zero-shot setting, using only feature names and task descriptions, as a viable alternative to traditional approaches. We evaluate multiple LLMs (GPT-5.0, GPT-4.0, Gemini-2.5 etc.) on the EMBOD dataset (a fusion of EMBER and BODMAS benchmark datasets), comparing them against established FS methods across several classifiers, including Random Forest, Extra Trees, MLP, and KNN. Performance is assessed using accuracy, precision, recall, F1, AUC, MCC, and runtime. Our results demonstrate that LLM-guided zero-shot feature selection achieves competitive performance with traditional FS methods while offering additional advantages in interpretability, stability, and reduced dependence on labeled data. These findings position zero-shot LLM-based FS as a promising alternative strategy for effective and interpretable malware detection, paving the way for knowledge-guided feature selection in security-critical applications
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09629v1">Stop Testing Attacks, Start Diagnosing Defenses: The Four-Checkpoint Framework Reveals Where LLM Safety Breaks</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 17 pages, pre-print
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deploy safety mechanisms to prevent harmful outputs, yet these defenses remain vulnerable to adversarial prompts. While existing research demonstrates that jailbreak attacks succeed, it does not explain \textit{where} defenses fail or \textit{why}. To address this gap, we propose that LLM safety operates as a sequential pipeline with distinct checkpoints. We introduce the \textbf{Four-Checkpoint Framework}, which organizes safety mechanisms along two dimensions: processing stage (input vs.\ output) and detection level (literal vs.\ intent). This creates four checkpoints, CP1 through CP4, each representing a defensive layer that can be independently evaluated. We design 13 evasion techniques, each targeting a specific checkpoint, enabling controlled testing of individual defensive layers. Using this framework, we evaluate GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across 3,312 single-turn, black-box test cases. We employ an LLM-as-judge approach for response classification and introduce Weighted Attack Success Rate (WASR), a severity-adjusted metric that captures partial information leakage overlooked by binary evaluation. Our evaluation reveals clear patterns. Traditional Binary ASR reports 22.6\% attack success. However, WASR reveals 52.7\%, a 2.3$\times$ higher vulnerability. Output-stage defenses (CP3, CP4) prove weakest at 72--79\% WASR, while input-literal defenses (CP1) are strongest at 13\% WASR. Claude achieves the strongest safety (42.8\% WASR), followed by GPT-5 (55.9\%) and Gemini (59.5\%). These findings suggest that current defenses are strongest at input-literal checkpoints but remain vulnerable to intent-level manipulation and output-stage techniques. The Four-Checkpoint Framework provides a structured approach for identifying and addressing safety vulnerabilities in deployed systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10936v3">Cochain: Balancing Insufficient and Excessive Collaboration in LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 35 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating the collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves the business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09624v1">MILE-RefHumEval: A Reference-Free, Multi-Independent LLM Framework for Human-Aligned Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We introduce MILE-RefHumEval, a reference-free framework for evaluating Large Language Models (LLMs) without ground-truth annotations or evaluator coordination. It leverages an ensemble of independently prompted evaluators guided by a human-aligned schema, supporting both discrete and continuous scoring judgement. With task-specific prompts from best candidate selection, summarization and image captioning to dialogue, MILE-RefHumEval provides flexible, interpretable, and scalable assessments. Experiments show it aligns closely with human judgments, outperforms prior methods, and reduces computational overhead, offering an efficient, robust, and human-aligned solution for real-world LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08060v2">Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted to AccML@HiPEAC 2026
    </div>
    <details class="paper-abstract">
      LLM deployment on resource-constrained edge devices faces severe latency constraints, particularly in real-time applications where delayed responses can compromise safety or usability. Among many approaches to mitigate the inefficiencies of sequential token-by-token generation, Speculative Decoding (SD) has emerged as a promising technique. However, SD at the edge is hindered by two major challenges: (1) integrating SD into a compiler-based workflow without sacrificing performance or programmability, and (2) exploiting the heterogeneous compute resources of modern SoCs through carefully designed partitioning strategies. This work addresses these challenges by using an analytical cost model that explores heterogeneous hardware configurations and guides coarse-grained partitioning of LLM subgraphs, particularly with edge-typical short input sequence lengths. The cost model predicts when speculative sampling and heterogeneous execution are jointly beneficial and is validated on an edge device featuring a hexacore Cortex-A CPU and a Mali GPU, revealing up to 1.68$\times$ speedup for translation tasks, closely matching analytic expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09598v1">Learning from the Irrecoverable: Error-Localized Policy Optimization for Tool-Integrated LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Tool-integrated reasoning (TIR) enables LLM agents to solve tasks through planning, tool use, and iterative revision, but outcome-only reinforcement learning in this setting suffers from sparse, delayed rewards and weak step-level credit assignment. In long-horizon TIR trajectories, an early irrecoverable mistake can determine success or failure, making it crucial to localize the first irrecoverable step and leverage it for fine-grained credit assignment. We propose Error-Localized Policy Optimization (ELPO), which localizes the first irrecoverable step via binary-search rollout trees under a fixed rollout budget, converts the resulting tree into stable learning signals through hierarchical advantage attribution, and applies error-localized adaptive clipping to strengthen corrective updates on the critical step and its suffix. Across TIR benchmarks in math, science QA, and code execution, ELPO consistently outperforms strong Agentic RL baselines under comparable sampling budgets, with additional gains in Pass@K and Major@K scaling, rollout ranking quality, and tool-call efficiency. Our code will be publicly released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09578v1">Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Despite algorithm-level innovations for multi-agent reinforcement learning (MARL), the underlying networked infrastructure for large-scale MARL training remains underexplored. Existing training frameworks primarily optimize for single-agent scenarios and fail to address the unique system-level challenges of MARL, including rollout-training synchronization barriers, rollout load imbalance, and training resource underutilization. To bridge this gap, we propose FlexMARL, the first end-to-end training framework that holistically optimizes rollout, training, and their orchestration for large-scale LLM-based MARL. Specifically, FlexMARL introduces the joint orchestrator to manage data flow under the rollout-training disaggregated architecture. Building upon the experience store, a novel micro-batch driven asynchronous pipeline eliminates the synchronization barriers while providing strong consistency guarantees. Rollout engine adopts a parallel sampling scheme combined with hierarchical load balancing, which adapts to skewed inter/intra-agent request patterns. Training engine achieves on-demand hardware binding through agent-centric resource allocation. The training states of different agents are swapped via unified and location-agnostic communication. Empirical results on a large-scale production cluster demonstrate that FlexMARL achieves up to 7.3x speedup and improves hardware utilization by up to 5.6x compared to existing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09574v1">Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Tree-search decoding is an effective form of test-time scaling for large language models (LLMs), but real-world deployment imposes a fixed per-query token budget that varies across settings. Existing tree-search policies are largely budget-agnostic, treating the budget as a termination condition, which can lead to late-stage over-branching or premature termination. We propose {Budget-Guided MCTS} (BG-MCTS), a tree-search decoding algorithm that aligns its search policy with the remaining token budget: it starts with broad exploration, then prioritizes refinement and answer completion as the budget depletes while reducing late-stage branching from shallow nodes. BG-MCTS consistently outperforms budget-agnostic tree-search baselines across different budgets on MATH500 and AIME24/25 with open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11097v2">The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07543v2">MSP-LLM: A Unified Large Language Model Framework for Complete Material Synthesis Planning</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Material synthesis planning (MSP) remains a fundamental and underexplored bottleneck in AI-driven materials discovery, as it requires not only identifying suitable precursor materials but also designing coherent sequences of synthesis operations to realize a target material. Although several AI-based approaches have been proposed to address isolated subtasks of MSP, a unified methodology for solving the entire MSP task has yet to be established. We propose MSP-LLM, a unified LLM-based framework that formulates MSP as a structured process composed of two constituent subproblems: precursor prediction (PP) and synthesis operation prediction (SOP). Our approach introduces a discrete material class as an intermediate decision variable that organizes both tasks into a chemically consistent decision chain. For OP, we further incorporate hierarchical precursor types as synthesis-relevant inductive biases and employ an explicit conditioning strategy that preserves precursor-related information in the autoregressive decoding state. Extensive experiments show that MSP-LLM consistently outperforms existing methods on both PP and SOP, as well as on the complete MSP task, demonstrating an effective and scalable framework for MSP that can accelerate real-world materials discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.16276v4">Mechanism Design for LLM Fine-tuning with Multiple Reward Models</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to aggregate multiple preferences has attracted considerable research attention. With aggregation algorithms advancing, a potential economic scenario arises where fine-tuning services are provided to agents with different preferences. In this context, agents may benefit from strategically misreporting their preferences, but this could harm the aggregation performance. This paper addresses such incentive issues by framing it as a mechanism design problem: an LLM provider determines the fine-tuning objective (training rule) and the pricing scheme (payment rule) for agents. We primarily focus on training rules that maximize social welfare subject to certain regularizations, referred to as SW-Max rules. First, we show that under most circumstances, truthful reporting is sub-optimal with simply a SW-Max rule, thereby highlighting the necessity of payments. Second, we extend the VCG payment to implement SW-Max rules in dominant-strategy incentive compatibility (DSIC). We characterize sufficient conditions for payment equivalence and derive the necessary conditions for a payment rule to implement a SW-Max rule in DSIC and other principles. Third, we demonstrate that our mechanism is approximately DSIC with perturbed input, showcasing its robustness against the inevitable errors in real-world applications. Experiments on real LLM training results further confirm the practical implications of our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17684v2">Can LLMs Automate Fact-Checking Article Writing?</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted to TACL 2026, pre-MIT Press publication version
    </div>
    <details class="paper-abstract">
      Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. In particular, we argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction. The code for our implementation is available at https://github.com/mbzuai-nlp/qraft.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17653v2">GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Geometric spatial reasoning forms the foundation of many applications in artificial intelligence, yet the ability of large language models (LLMs) to operate over geometric spatial information expressed in procedural code remains underexplored. In this paper, we address this gap by formalizing the Program-to-Geometry task, which challenges models to translate programmatic drawing code into accurate and abstract geometric reasoning. To evaluate this capability, we present GeoGramBench, a benchmark of 500 carefully refined problems organized by a tailored three-level taxonomy that considers geometric complexity rather than traditional mathematical reasoning complexity. Our comprehensive evaluation of 17 frontier LLMs reveals consistent and pronounced deficiencies: even the most advanced models achieve less than 50% accuracy at the highest abstraction level. These results highlight the unique challenges posed by program-driven spatial reasoning and establish GeoGramBench as a valuable resource for advancing research in symbolic-to-spatial geometric reasoning. Project page: https://github.com/LiAuto-DSR/GeoGramBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05946v1">$f$-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Recent research shows that Preference Alignment (PA) objectives act as divergence estimators between aligned (chosen) and unaligned (rejected) response distributions. In this work, we extend this divergence-based perspective to general alignment settings, such as reinforcement learning with verifiable rewards (RLVR), where only environmental rewards are available. Within this unified framework, we propose $f$-Group Relative Policy Optimization ($f$-GRPO), a class of on-policy reinforcement learning, and $f$-Hybrid Alignment Loss ($f$-HAL), a hybrid on/off policy objectives, for general LLM alignment based on variational representation of $f$-divergences. We provide theoretical guarantees that these classes of objectives improve the average reward after alignment. Empirically, we validate our framework on both RLVR (Math Reasoning) and PA tasks (Safety Alignment), demonstrating superior performance and flexibility compared to current methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05945v1">AgenticTagger: Structured Item Representation for Recommendation with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      High-quality representations are a core requirement for effective recommendation. In this work, we study the problem of LLM-based descriptor generation, i.e., keyphrase-like natural language item representation generation frameworks with minimal constraints on downstream applications. We propose AgenticTagger, a framework that queries LLMs for representing items with sequences of text descriptors. However, open-ended generation provides little control over the generation space, leading to high cardinality, low-performance descriptors that renders downstream modeling challenging. To this end, AgenticTagger features two core stages: (1) a vocabulary building stage where a set of hierarchical, low-cardinality, and high-quality descriptors is identified, and (2) a vocabulary assignment stage where LLMs assign in-vocabulary descriptors to items. To effectively and efficiently ground vocabulary in the item corpus of interest, we design a multi-agent reflection mechanism where an architect LLM iteratively refines the vocabulary guided by parallelized feedback from annotator LLMs that validates the vocabulary against item data. Experiments on public and private data show AgenticTagger brings consistent improvements across diverse recommendation scenarios, including generative and term-based retrieval, ranking, and controllability-oriented, critique-based recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04856v2">CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      From generating headlines to fabricating news, the Large Language Models (LLMs) are typically assessed by their final outputs, under the safety assumption that a refusal response signifies safe reasoning throughout the entire process. Challenging this assumption, our study reveals that during fake news generation, even when a model rejects a harmful request, its Chain-of-Thought (CoT) reasoning may still internally contain and propagate unsafe narratives. To analyze this phenomenon, we introduce a unified safety-analysis framework that systematically deconstructs CoT generation across model layers and evaluates the role of individual attention heads through Jacobian-based spectral metrics. Within this framework, we introduce three interpretable measures: stability, geometry, and energy to quantify how specific attention heads respond or embed deceptive reasoning patterns. Extensive experiments on multiple reasoning-oriented LLMs show that the generation risk rise significantly when the thinking mode is activated, where the critical routing decisions concentrated in only a few contiguous mid-depth layers. By precisely identifying the attention heads responsible for this divergence, our work challenges the assumption that refusal implies safety and provides a new understanding perspective for mitigating latent reasoning risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05933v1">Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Policy mirror descent (PMD) provides a principled framework for reinforcement learning (RL) by iteratively solving KL-regularized policy improvement subproblems. While this approach has been adopted in training advanced LLMs such as Kimi K1.5/K2, the ideal closed-form PMD updates require reliable partition function estimation, a significant challenge when working with limited rollouts in the vast action spaces of LLMs. We investigate a practical algorithm, termed PMD-mean, that approximates the log-partition term with the mean reward under the sampling policy and performs regression in log-policy space. Specifically, we characterize the population solution of PMD-mean and demonstrate that it implicitly optimizes mirror descent subproblems with an adaptive mixed KL--$χ^2$ regularizer. This additional $χ^2$ regularization constrains large probability changes, producing more conservative updates when expected rewards are low and enhancing robustness against finite-sample estimation errors. Experiments on math reasoning tasks show that PMD-mean achieves superior performance with improved stability and time efficiency. These findings deepen our understanding of PMD-mean and illuminate pathways toward principled improvements in RL algorithms for LLMs. Code is available at https://github.com/horizon-rl/OpenKimi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05932v1">Polyglots or Multitudes? Multilingual LLM Answers to Value-laden Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 17 pages, 5 figures (8 pages of references and appendices)
    </div>
    <details class="paper-abstract">
      Multiple-Choice Questions (MCQs) are often used to assess knowledge, reasoning abilities, and even values encoded in large language models (LLMs). While the effect of multilingualism has been studied on LLM factual recall, this paper seeks to investigate the less explored question of language-induced variation in value-laden MCQ responses. Are multilingual LLMs consistent in their responses across languages, i.e. behave like theoretical polyglots, or do they answer value-laden MCQs depending on the language of the question, like a multitude of monolingual models expressing different values through a single model? We release a new corpus, the Multilingual European Value Survey (MEVS), which, unlike prior work relying on machine translation or ad hoc prompts, solely comprises human-translated survey questions aligned in 8 European languages. We administer a subset of those questions to over thirty multilingual LLMs of various sizes, manufacturers and alignment-fine-tuning status under comprehensive, controlled prompt variations including answer order, symbol type, and tail character. Our results show that while larger, instruction-tuned models display higher overall consistency, the robustness of their responses varies greatly across questions, with certain MCQs eliciting total agreement within and across models while others leave LLM answers split. Language-specific behavior seems to arise in all consistent, instruction-fine-tuned models, but only on certain questions, warranting a further study of the selective effect of preference fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05929v1">KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language models rely on kv-caches to avoid redundant computation during autoregressive decoding, but as context length grows, reading and writing the cache can quickly saturate GPU memory bandwidth. Recent work has explored KV-cache compression, yet most approaches neglect the data-dependent nature of kv-caches and their variation across layers. We introduce KV-CoRE KV-cache Compressibility by Rank Evaluation), an SVD-based method for quantifying the data-dependent low-rank compressibility of kv-caches. KV-CoRE computes the optimal low-rank approximation under the Frobenius norm and, being gradient-free and incremental, enables efficient dataset-level, layer-wise evaluation. Using this method, we analyze multiple models and datasets spanning five English domains and sixteen languages, uncovering systematic patterns that link compressibility to model architecture, training data, and language coverage. As part of this analysis, we employ the Normalized Effective Rank as a metric of compressibility and show that it correlates strongly with performance degradation under compression. Our study establishes a principled evaluation framework and the first large-scale benchmark of kv-cache compressibility in LLMs, offering insights for dynamic, data-aware compression and data-centric model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20295v4">SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      The common approach to communicate a large language model's (LLM) uncertainty is to add a percentage number or a hedging word to its response. But is this all we can do? Instead of generating a single answer and then hedging it, an LLM that is fully transparent to the user needs to be able to reflect on its internal belief distribution and output a summary of all options it deems possible, and how likely they are. To test whether LLMs possess this capability, we develop the SelfReflect metric, an information-theoretic distance between a given summary and a distribution over answers. In interventional and human studies, we find that SelfReflect indicates even slight deviations, yielding a fine measure of faithfulness between a summary string and an LLM's actual internal distribution over answers. With SelfReflect, we make a resounding negative observation: modern LLMs are, across the board, incapable of revealing what they are uncertain about, neither through reasoning, nor chains-of-thoughts, nor explicit finetuning. However, we do find that LLMs are able to generate faithful summaries of their uncertainties if we help them by sampling multiple outputs and feeding them back into the context. This simple approach shines a light at the universal way of communicating LLM uncertainties whose future development the SelfReflect score enables. To support the development of this universal form of LLM uncertainties, we publish the code that implements our metric for arbitrary LLMs under https://github.com/apple/ml-selfreflect .
    </details>
</div>
