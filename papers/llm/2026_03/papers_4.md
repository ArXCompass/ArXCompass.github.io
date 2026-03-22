# llm - 2026_03

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11103v1">Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have achieved remarkable success in code generation, they often struggle with the deep, long-horizon reasoning required for complex software engineering. We attribute this limitation to the nature of standard pre-training data: static software repositories represent only the terminal state of an intricate intellectual process, abstracting away the intermediate planning, debugging, and iterative refinement. To bridge this gap, we propose a novel paradigm: understanding via reconstruction. We hypothesize that reverse-engineering the latent agentic trajectories -- the planning, reasoning, and debugging steps -- behind static repositories provides a far richer supervision signal than raw code alone. To operationalize this, we introduce a framework that synthesizes these trajectories using a multi-agent simulation. This process is grounded in the structural realities of the source repositories (e.g., dependency graphs and file hierarchies) to ensure fidelity. Furthermore, to guarantee the logical rigor of the synthetic data, we employ a search-based optimization technique that iteratively refines the Chain-of-Thought (CoT) reasoning to maximize the likelihood of the ground-truth code. Empirical results demonstrate that continuous pre-training on these reconstructed trajectories significantly enhances Llama-3-8B's performance across diverse benchmarks, including long-context understanding, coding proficiency, and agentic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10560v1">PET-F2I: A Comprehensive Benchmark and Parameter-Efficient Fine-Tuning of LLMs for PET/CT Report Impression Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      PET/CT imaging is pivotal in oncology and nuclear medicine, yet summarizing complex findings into precise diagnostic impressions is labor-intensive. While LLMs have shown promise in medical text generation, their capability in the highly specialized domain of PET/CT remains underexplored. We introduce PET-F2I-41K (PET Findings-to-Impression Benchmark), a large-scale benchmark for PET/CT impression generation using LLMs, constructed from over 41k real-world reports. Using PET-F2I-41K, we conduct a comprehensive evaluation of 27 models across proprietary frontier LLMs, open-source generalist models, and medical-domain LLMs, and we develop a domain-adapted 7B model (PET-F2I-7B) fine-tuned from Qwen2.5-7B-Instruct via LoRA. Beyond standard NLG metrics (e.g., BLEU-4, ROUGE-L, BERTScore), we propose three clinically grounded metrics - Entity Coverage Rate (ECR), Uncovered Entity Rate (UER), and Factual Consistency Rate (FCR) - to assess diagnostic completeness and factual reliability. Experiments reveal that neither frontier nor medical-domain LLMs perform adequately in zero-shot settings. In contrast, PET-F2I-7B achieves substantial gains (e.g., 0.708 BLEU-4) and a 3.0x improvement in entity coverage over the strongest baseline, while offering advantages in cost, latency, and privacy. Beyond this modeling contribution, PET-F2I-41K establishes a standardized evaluation framework to accelerate the development of reliable and clinically deployable reporting systems for PET/CT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20508v2">Assessing the Political Fairness of Multilingual LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Accepted at LREC 2026. Added results with new models and two-ANOVA. Same conclusions
    </div>
    <details class="paper-abstract">
      The political biases of Large Language Models (LLMs) are usually assessed by simulating their answers to English surveys. In this work, we propose an alternative framing of political biases, relying on principles of fairness in multilingual translation. We systematically compare the translation quality of speeches in the European Parliament (EP), observing systematic differences with majority parties from left and right being better translated than outsider parties. This study is made possible by a new, 21-way multiparallel version of EuroParl, the parliamentary proceedings of the EP, which includes the political affiliations of each speaker. The dataset consists of 1.5M sentences for a total of 40M words and 249M characters. It covers three years, 1000+ speakers, 7 countries, 12 EU parties, 25 EU committees, and hundreds of national parties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10521v1">IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Instruction hierarchy (IH) defines how LLMs prioritize system, developer, user, and tool instructions under conflict, providing a concrete, trust-ordered policy for resolving instruction conflicts. IH is key to defending against jailbreaks, system prompt extractions, and agentic prompt injections. However, robust IH behavior is difficult to train: IH failures can be confounded with instruction-following failures, conflicts can be nuanced, and models can learn shortcuts such as overrefusing. We introduce IH-Challenge, a reinforcement learning training dataset, to address these difficulties. Fine-tuning GPT-5-Mini on IH-Challenge with online adversarial example generation improves IH robustness by +10.0% on average across 16 in-distribution, out-of-distribution, and human red-teaming benchmarks (84.1% to 94.1%), reduces unsafe behavior from 6.6% to 0.7% while improving helpfulness on general safety evaluations, and saturates an internal static agentic prompt injection evaluation, with minimal capability regression. We release the IH-Challenge dataset (https://huggingface.co/datasets/openai/ih-challenge) to support future research on robust instruction hierarchy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10955v2">Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      The agent--tool interaction loop is a critical attack surface for modern Large Language Model (LLM) agents. Existing denial-of-service (DoS) attacks typically function at the user-prompt or retrieval-augmented generation (RAG) context layer and are inherently single-turn in nature. This limitation restricts cost amplification and diminishes stealth in goal-oriented workflows. To address these issues, we proposed a stealthy, multi-turn economic DoS attack at the tool layer under the Model Context Protocol (MCP). By simply editing text-visible fields and implementing a template-driven return policy, our malicious server preserves function signatures and the terminal benign payload while steering agents into prolonged, verbose tool-calling chains. We optimize these text-only edits with Monte Carlo Tree Search (MCTS) to maximize cost under a task-success constraint. Across six LLMs on ToolBench and BFCL benchmarks, our attack yields trajectories over 60K tokens, increases per-query cost by up to 658 times, raises energy by 100 to 560 times, and pushes GPU key-value (KV) cache occupancy to 35--74%. Standard prompt filters and output trajectory monitors seldom detect these attacks, highlighting the need for defenses that safeguard agentic processes rather than focusing solely on final outcomes. We will release the code soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10476v1">Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) has progressed substantially in single-agent settings through paradigms such as RLHF and Constitutional AI, with recent work exploring scalable alternatives such as RLAIF and evolving alignment objectives. However, these approaches remain limited in multi-stakeholder settings, where conflicting values arise and deliberative negotiation capabilities are required. This work proposes a multi-agent negotiation-based alignment framework that aligns LLMs to Collective Agency (CA)-an existing alignment objective introduced to promote the continual expansion of agency-while simultaneously improving conflict-resolution capability. To enable scalable training, two self-play instances of the same LLM, assigned opposing personas, engage in structured turn-based dialogue to synthesize mutually beneficial solutions. We generate synthetic moral-dilemma prompts and conflicting persona pairs, and optimize the policy via RLAIF using GRPO with an external LLM reward model. While rewards are computed from CA scores assigned to the final completion, gradients are applied to dialogue tokens to directly improve deliberative interaction dynamics. Experiments show that the resulting model achieves CA alignment comparable to a single-agent baseline while substantially improving conflict-resolution performance without degrading general language capabilities. These results suggest that negotiation-driven deliberation training provides a practical path toward LLMs that better support collective decision-making in value-conflict scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03195v4">From Text to Alpha: Can LLMs Track Evolving Signals in Corporate Disclosures?</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Natural language processing (NLP) has been widely used in quantitative finance, but traditional methods often struggle to capture rich narratives in corporate disclosures, leaving potentially informative signals under-explored. Large language models (LLMs) offer a promising alternative due to their ability to extract nuanced semantics. In this paper, we ask whether semantic signals extracted by LLMs from corporate disclosures predict alpha, defined as abnormal returns beyond broad market movements and common risk factors. We introduce a simple framework, LLM as extractor, embedding as ruler, which extracts context-aware, metric-focused textual spans and quantifies semantic changes across consecutive disclosure periods using embedding-based similarity. This allows us to measure the degree of metric shifting -- how much firms move away from previously emphasized metrics, referred as moving targets. In experiments with portfolio and cross-sectional regression tests against a recent NER-based baseline, our method achieves more than twice the risk-adjusted alpha and shows significantly stronger predictive power. Qualitative analysis suggests that these gains stem from preserving contextual qualifiers and filtering out non-metric terms that keyword-based approaches often miss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10444v1">The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Large language models trained on natural language exhibit pronounced anisotropy: a small number of directions concentrate disproportionate energy, while the remaining dimensions form a broad semantic tail. In low-bit training regimes, this geometry becomes numerically unstable. Because blockwise quantization scales are determined by extreme elementwise magnitudes, dominant directions stretch the dynamic range, compressing long-tail semantic variation into narrow numerical bins. We show that this instability is primarily driven by a coherent rank-one mean bias, which constitutes the dominant component of spectral anisotropy in LLM representations. This mean component emerges systematically across layers and training stages and accounts for the majority of extreme activation magnitudes, making it the principal driver of dynamic-range inflation under low precision. Crucially, because the dominant instability is rank-one, it can be eliminated through a simple source-level mean-subtraction operation. This bias-centric conditioning recovers most of the stability benefits of SVD-based spectral methods while requiring only reduction operations and standard quantization kernels. Empirical results on FP4 (W4A4G4) training show that mean removal substantially narrows the loss gap to BF16 and restores downstream performance, providing a hardware-efficient path to stable low-bit LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01968v3">Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant, uninformative, or even harmful. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves downstream performance. Code is available at https://github.com/UCSC-REAL/TokenCleaning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11082v1">Quality-Driven Agentic Reasoning for LLM-Assisted Software Design: Questions-of-Thoughts (QoT) as a Time-Series Self-QA Chain</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated AI-assisted software development, yet practical deployment remains constrained by incomplete implementations, weak modularization, and inconsistent security practices. We introduce Questions-of-Thoughts (QoT), a quality-driven inference-time scaffold that turns a user goal into (i) an ordered sequence of engineering steps and (ii) stepwise self-questioning to verify constraints and reduce omission errors, while maintaining a lightweight reasoning record that stabilizes subsequent design decisions. We evaluate QoT across three representative backend engineering domains: API Design, Data Communication, and File Systems. Each task requires multi-module decomposition and exposes standard failure modes in LLM-generated systems. To enable data-driven comparison, we score generated artifacts using an ISO/IEC-inspired quality rubric that measures Scalability, Completeness, Modularity, and Security. We report domain-wise gains as the change in total quality score, defined as the QoT score minus the NoQoT score. Results show capacity-dependent improvements: QoT yields consistent quality improvements for larger models and more complex domains, while smaller models may exhibit trade-offs under tight context and planning budgets. We release an open artifact with prompts, scoring guidelines, raw generations, and scripts that reproduce the reported tables and figures to support applied AI and data analytics research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10249v1">DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 22 pages, including supplemental material. 9 Figures
    </div>
    <details class="paper-abstract">
      Engineering analysis automation in product development relies on rigid interfaces between tools, data formats and documented processes. When these interfaces change, as they routinely do as the product evolves in the engineering ecosystem, the automation support breaks. This paper presents a DUCTILE (Delegated, User-supervised Coordination of Tool- and document-Integrated LLM-Enabled) agentic orchestration, an approach for developing, executing and evaluating LLM-based agentic automation support of engineering analysis tasks. The approach separates adaptive orchestration, performed by the LLM agent, from deterministic execution, performed by verified engineering tools. The agent interprets documented design practices, inspects input data and adapts the processing path, while the engineer supervises and exercises final judgment. DUCTILE is demonstrated on an industrial structural analysis task at an aerospace manufacturer, where the agent handled input deviations in format, units, naming conventions and methodology that would break traditional scripted pipelines. Evaluation against expert-defined acceptance criteria and deployment with practicing engineers confirm that the approach produces correct, methodologically compliant results across repeated independent runs. The paper discusses practical consequences of adopting agentic automation, including unintended effects on the nature of engineering work and the tension between removing mundane tasks and creating an exhausting supervisory role.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09723v4">AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents Amazon.com, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12918v2">ThinkPatterns-21k: A Systematic Study on the Impact of Thinking Patterns in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated enhanced performance through the \textit{Thinking then Responding} paradigm, where models generate internal thoughts before final responses (aka, System 2 thinking). However, existing research lacks a systematic understanding of the mechanisms underlying how thinking patterns affect performance across model sizes. In this work, we conduct a comprehensive analysis of the impact of various thinking types on model performance and introduce ThinkPatterns-21k, a curated dataset comprising 21k instruction-response pairs (QA) collected from existing instruction-following datasets with five thinking types. For each pair, we augment it with five distinct internal thinking patterns: one unstructured thinking (monologue) and four structured variants (decomposition, self-ask, self-debate and self-critic), while maintaining the same instruction and response. Through extensive evaluation across different model sizes (3B-32B parameters), we have two key findings: (1) smaller models (<30B parameters) can benefit from most of structured thinking patterns, while larger models (32B) with structured thinking like decomposition would degrade performance and (2) unstructured monologue demonstrates broad effectiveness across different model sizes. Finally, we released all of our datasets, checkpoints, training logs of diverse thinking patterns to reproducibility, aiming to facilitate further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.21735v4">GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Ensuring reliable data-driven decisions is crucial in domains where analytical accuracy directly impacts safety, compliance, or operational outcomes. Decision support in such domains relies on large tabular datasets, where manual analysis is slow, costly, and error-prone. While Large Language Models (LLMs) offer promising automation potential, they face challenges in analytical reasoning, structured data handling, and ambiguity resolution. This paper introduces GateLens, an LLM-based architecture for reliable analysis of complex tabular data. Its key innovation is the use of Relational Algebra (RA) as a formal intermediate representation between natural-language reasoning and executable code, addressing the reasoning-to-code gap that can arise in direct generation approaches. In our automotive instantiation, GateLens translates natural language queries into RA expressions and generates optimized Python code. Unlike traditional multi-agent or planning-based systems that can be slow, opaque, and costly to maintain, GateLens emphasizes speed, transparency, and reliability. We validate the architecture in automotive software release analytics, where experimental results show that GateLens outperforms the existing Chain-of-Thought (CoT) + Self-Consistency (SC) based system on real-world datasets, particularly in handling complex and ambiguous queries. Ablation studies confirm the essential role of the RA layer. Industrial deployment demonstrates over 80% reduction in analysis time while maintaining high accuracy across domain-specific tasks. GateLens operates effectively in zero-shot settings without requiring few-shot examples or agent orchestration. This work advances deployable LLM system design by identifying key architectural features--intermediate formal representations, execution efficiency, and low configuration overhead--crucial for domain-specific analytical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.08999v3">MCP Bridge: A Lightweight, LLM-Agnostic RESTful Proxy for Model Context Protocol Servers</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 42 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly augmented with external tools through standardized interfaces like the Model Context Protocol (MCP). However, current MCP implementations face critical limitations: they typically require local process execution through STDIO transports, making them impractical for resource-constrained environments like mobile devices, web browsers, and edge computing. We present MCP Bridge, a lightweight RESTful proxy that connects to multiple MCP servers and exposes their capabilities through a unified API. Unlike existing solutions, MCP Bridge is fully LLM-agnostic, supporting any backend regardless of vendor. The system implements a risk-based execution model with three security levels-standard execution, confirmation workflow, and Docker isolation-while maintaining backward compatibility with standard MCP clients. However, reliable execution within this framework requires models that can strictly adhere to protocol schemas. To this end, we also fine-tuned the Qwen3 4B and 8B model family on the Agent-Ark/Toucan-1.5M dataset using four Reinforcement Learning techniques: Group Relative Policy Optimization (GRPO), Dr. GRPO, Beta Normalization Policy Optimization (BNPO), and Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO). Evaluated on the MCPToolBench++ benchmark, our optimized model achieves an F1 score of 73.0% that outperforms GPT-OSS-120B (62.17%) and remains competitive with the 70B+ parameter baselines. Evaluation demonstrates that MCP Bridge successfully addresses the constraints of direct MCP connections while providing enhanced security controls and cross-platform compatibility, enabling sophisticated LLM-powered applications in previously inaccessible environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09906v1">Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      While reasoning in LLMs plays a natural role in math, code generation, and multi-hop factual questions, its effect on simple, single-hop factual questions remains unclear. Such questions do not require step-by-step logical decomposition, making the utility of reasoning highly counterintuitive. Nevertheless, we find that enabling reasoning substantially expands the capability boundary of the model's parametric knowledge recall, unlocking correct answers that are otherwise effectively unreachable. Why does reasoning aid parametric knowledge recall when there are no complex reasoning steps to be done? To answer this, we design a series of hypothesis-driven controlled experiments, and identify two key driving mechanisms: (1) a computational buffer effect, where the model uses the generated reasoning tokens to perform latent computation independent of their semantic content; and (2) factual priming, where generating topically related facts acts as a semantic bridge that facilitates correct answer retrieval. Importantly, this latter generative self-retrieval mechanism carries inherent risks: we demonstrate that hallucinating intermediate facts during reasoning increases the likelihood of hallucinations in the final answer. Finally, we show that our insights can be harnessed to directly improve model accuracy by prioritizing reasoning trajectories that contain hallucination-free factual statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.00957v4">Generative AI and LLMs in Industry: A text-mining Analysis and Critical Evaluation of Guidelines and Policy Statements Across Fourteen Industrial Sectors</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      The rise of Generative AI (GAI) and Large Language Models (LLMs) has transformed industrial landscapes, offering unprecedented opportunities for efficiency and innovation while raising critical ethical, regulatory, and operational challenges. This study conducts a text-based analysis of 160 guidelines and policy statements across fourteen industrial sectors, utilizing systematic methods and text-mining techniques to evaluate the governance of these technologies. By examining global directives, industry practices, and sector-specific policies, the paper highlights the complexities of balancing innovation with ethical accountability and equitable access. The findings provide actionable insights and recommendations for fostering responsible, transparent, and safe integration of GAI and LLMs in diverse industry contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09892v1">MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Continual fine-tuning of large language models (LLMs) is becoming increasingly crucial as these models are deployed in dynamic environments where tasks and data distributions evolve over time. While strong adaptability enables rapid acquisition of new knowledge, it also exposes LLMs to catastrophic forgetting, where previously learned skills degrade during sequential training. Existing replay-based strategies, such as fixed interleaved replay, accuracy-supervised, and loss-driven scheduling, remain limited: some depend on heuristic rules and provide only partial mitigation of forgetting, while others improve performance but incur substantial computational overhead. Motivated by retention dynamics under sequential fine-tuning, we propose Memory-Inspired Sampler and Scheduler Replay (MSSR), an experience replay framework that estimates sample-level memory strength and schedules rehearsal at adaptive intervals to mitigate catastrophic forgetting while maintaining fast adaptation. Extensive experiments across three backbone models and 11 sequential tasks show that MSSR consistently outperforms state-of-the-art replay baselines, with particularly strong gains on reasoning-intensive and multiple-choice benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09890v1">Influencing LLM Multi-Agent Dialogue via Policy-Parameterized Prompts</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a new paradigm for multi-agent systems. However, existing research on the behaviour of LLM-based multi-agents relies on ad hoc prompts and lacks a principled policy perspective. Different from reinforcement learning, we investigate whether prompt-as-action can be parameterized so as to construct a lightweight policy which consists of a sequence of state-action pairs to influence conversational behaviours without training. Our framework regards prompts as actions executed by LLMs, and dynamically constructs prompts through five components based on the current state of the agent. To test the effectiveness of parameterized control, we evaluated the dialogue flow based on five indicators: responsiveness, rebuttal, evidence usage, non-repetition, and stance shift. We conduct experiments using different LLM-driven agents in two discussion scenarios related to the general public and show that prompt parameterization can influence the dialogue dynamics. This result shows that policy-parameterised prompts offer a simple and effective mechanism to influence the dialogue process, which will help the research of multi-agent systems in the direction of social simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08640v2">PostTrainBench: Can LLM Agents Automate LLM Post-Training?</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      AI agents have become surprisingly proficient at software engineering over the past year, largely due to improvements in reasoning capabilities. This raises a deeper question: can these systems extend their capabilities to automate AI research itself? In this paper, we explore post-training, the critical phase that turns base LLMs into useful assistants. We introduce PostTrainBench to benchmark how well LLM agents can perform post-training autonomously under bounded compute constraints (10 hours on one H100 GPU). We ask frontier agents (e.g., Claude Code with Opus 4.6) to optimize the performance of a base LLM on a particular benchmark (e.g., Qwen3-4B on AIME). Importantly, we do not provide any predefined strategies to the agents and instead give them full autonomy to find necessary information on the web, run experiments, and curate data. We find that frontier agents make substantial progress but generally lag behind instruction-tuned LLMs from leading providers: 23.2% for the best agent vs. 51.1% for official instruction-tuned models. However, agents can exceed instruction-tuned models in targeted scenarios: GPT-5.1 Codex Max achieves 89% on BFCL with Gemma-3-4B vs. 67% for the official model. We also observe several failure modes worth flagging. Agents sometimes engage in reward hacking: training on the test set, downloading existing instruction-tuned checkpoints instead of training their own, and using API keys they find to generate synthetic data without authorization. These behaviors are concerning and highlight the importance of careful sandboxing as these systems become more capable. Overall, we hope PostTrainBench will be useful for tracking progress in AI R&D automation and for studying the risks that come with it. Website and code are available at https://posttrainbench.com/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10091v1">Multi-Stream Perturbation Attack: Breaking Safety Alignment of Thinking LLMs Through Concurrent Task Interference</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      The widespread adoption of thinking mode in large language models (LLMs) has significantly enhanced complex task processing capabilities while introducing new security risks. When subjected to jailbreak attacks, the step-by-step reasoning process may cause models to generate more detailed harmful content. We observe that thinking mode exhibits unique vulnerabilities when processing interleaved multiple tasks. Based on this observation, we propose multi-stream perturbation attack, which generates superimposed interference by interweaving multiple task streams within a single prompt. We design three perturbation strategies: multi-stream interleaving, inversion perturbation, and shape transformation, which disrupt the thinking process through concurrent task interleaving, character reversal, and format constraints respectively. On JailbreakBench, AdvBench, and HarmBench datasets, our method achieves attack success rates exceeding most methods across mainstream models including Qwen3 series, DeepSeek, Qwen3-Max, and Gemini 2.5 Flash. Experiments show thinking collapse rates and response repetition rates reach up to 17% and 60% respectively, indicating multi-stream perturbation not only bypasses safety mechanisms but also causes thinking process collapse or repetitive outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09781v1">CLIOPATRA: Extracting Private Information from LLM Insights</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      As AI assistants become widely used, privacy-aware platforms like Anthropic's Clio have been introduced to generate insights from real-world AI use. Clio's privacy protections rely on layering multiple heuristic techniques together, including PII redaction, clustering, filtering, and LLM-based privacy auditing. In this paper, we put these claims to the test by presenting CLIOPATRA, the first privacy attack against "privacy-preserving" LLM insight systems. The attack involves a realistic adversary that carefully designs and inserts malicious chats into the system to break multiple layers of privacy protections and induce the leakage of sensitive information from a target user's chat. We evaluated CLIOPATRA on synthetically generated medical target chats, demonstrating that an adversary who knows only the basic demographics of a target user and a single symptom can successfully extract the user's medical history in 39% of cases by just inspecting Clio's output. Furthermore, CLIOPATRA can reach close to 100% when Clio is configured with other state-of-the-art models and the adversary's knowledge of the target user is increased. We also show that existing ad hoc mitigations, such as LLM-based privacy auditing, are unreliable and fail to detect major leaks. Our findings indicate that even when layered, current heuristic protections are insufficient to adequately protect user data in LLM-based analysis systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09701v1">An Empirical Study of Interaction Smells in Multi-Turn Human-LLM Collaborative Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation, evolving from static tools into dynamic conversational interfaces that facilitate complex, multi-turn collaborative programming. While LLMs exhibit remarkable proficiency in generating standalone code snippets, they often struggle to maintain contextual consistency during extended interactions, creating significant obstacles in the collaboration process. Existing benchmarks primarily emphasize the functional correctness of the final output, overlooking latent quality issues within the interaction process itself, which we term Interaction Smells. In this paper, we conduct an empirical study on sampled real-word user-LLM interactions from WildChat and LMSYS-Chat-1M datasets to systematically investigate Interaction Smells in human-LLM code generation tasks from the perspectives of phenomena, distribution, and mitigation. First, we establish the first taxonomy of Interaction Smells by manually performing open card sorting on real-world interaction logs. This taxonomy categorizes Interaction Smells into three primary categories, i.e., User Intent Quality, Historical Instruction Compliance, and Historical Response Violation, comprising nine specific subcategories. Next, we quantitatively evaluate six mainstream LLMs (i.e., GPT-4o, DeepSeek-Chat, Gemini 2.5, Qwen2.5-32B, Qwen2.5-72B, and Qwen3-235B-a22b) to analyze the distribution of Interaction Smells across different models. Finally, we propose Invariant-aware Constraint Evolution (InCE), a multi-agent framework designed to improve multi-turn interaction quality through explicit extraction of global invariants and pre-generation quality audits. Experimental results on the extended WildBench benchmark demonstrate that this lightweight mitigation approach significantly improves the Task Success Rate and effectively suppresses the occurrence of Interaction Smells.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18871v5">Periodic Asynchrony: An On-Policy Approach for Accelerating LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Since the introduction of the GRPO algorithm, reinforcement learning (RL) has attracted increasing attention for LLM post-training, yet training efficiency remains a critical challenge. In mainstream RL frameworks, inference and training are co-located on the same devices, and their synchronous execution prevents concurrent inference and training. In this work, we revisit the strategy of separating inference and training deployment, and propose a periodically asynchronous framework that transforms synchronous RL training into an asynchronous producer-consumer pipeline. Unlike existing asynchronous approaches that introduce off-policy bias, our design is provably equivalent to its synchronous counterpart, preserving strict on-policy correctness without any algorithmic modifications. We further introduce a unified tri-model architecture and a shared-prompt attention mechanism to support efficient asynchronous execution and reduce redundant computation. Experiments on NPU platforms demonstrate a three- to five-fold improvement in end-to-end training throughput over mainstream RL frameworks, while maintaining fully comparable accuracy, indicating its potential for widespread application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05494v2">Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large language models sometimes produce false or misleading responses. Two approaches to this problem are honesty elicitation -- modifying prompts or weights so that the model answers truthfully -- and lie detection -- classifying whether a given response is false. Prior work evaluates such methods on models specifically trained to lie or conceal information, but these artificial constructions may not resemble naturally-occurring dishonesty. We instead study open-weights LLMs from Chinese developers, which are trained to censor politically sensitive topics: Qwen3 models frequently produce falsehoods about subjects like Falun Gong or the Tiananmen protests while occasionally answering correctly, indicating they possess knowledge they are trained to suppress. Using this as a testbed, we evaluate a suite of elicitation and lie detection techniques. For honesty elicitation, sampling without a chat template, few-shot prompting, and fine-tuning on generic honesty data most reliably increase truthful responses. For lie detection, prompting the censored model to classify its own responses performs near an uncensored-model upper bound, and linear probes trained on unrelated data offer a cheaper alternative. The strongest honesty elicitation techniques also transfer to frontier open-weights models including DeepSeek R1. Notably, no technique fully eliminates false responses. We release all prompts, code, and transcripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09652v1">MiniAppBench: Evaluating the Shift from Text to Interactive HTML Responses in LLM-Powered Assistants</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs) in code generation, human-AI interaction is evolving from static text responses to dynamic, interactive HTML-based applications, which we term MiniApps. These applications require models to not only render visual interfaces but also construct customized interaction logic that adheres to real-world principles. However, existing benchmarks primarily focus on algorithmic correctness or static layout reconstruction, failing to capture the capabilities required for this new paradigm. To address this gap, we introduce MiniAppBench, the first comprehensive benchmark designed to evaluate principle-driven, interactive application generation. Sourced from a real-world application with 10M+ generations, MiniAppBench distills 500 tasks across six domains (e.g., Games, Science, and Tools). Furthermore, to tackle the challenge of evaluating open-ended interactions where no single ground truth exists, we propose MiniAppEval, an agentic evaluation framework. Leveraging browser automation, it performs human-like exploratory testing to systematically assess applications across three dimensions: Intention, Static, and Dynamic. Our experiments reveal that current LLMs still face significant challenges in generating high-quality MiniApps, while MiniAppEval demonstrates high alignment with human judgment, establishing a reliable standard for future research. Our code is available in github.com/MiniAppBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.01236v2">LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Cost-efficient path planning across multiple terrains is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes the overall travel cost. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments with limited opportunities for recharging or refueling. Despite its practical importance, cost-efficient path planning across heterogeneous terrains has received relatively limited attention in prior work. In this paper, we propose LLM-Advisor, a prompt-based, planner-agnostic framework that leverages large language models (LLMs) as non-decisive post-processing advisors for cost refinement, without modifying the underlying planner. While we observe that LLMs may occasionally produce implausible suggestions, we introduce two effective hallucination-mitigation strategies. We further introduce two datasets, MultiTerraPath and RUGD_v2, for systematic evaluation of cost-efficient path planning. Extensive experiments reveal that state-of-the-art LLMs, including GPT-4o, GPT-4-turbo, Gemini-2.5-Flash, and Claude-Opus-4, perform poorly in zero-shot terrain-aware path planning, highlighting their limited spatial reasoning capability. In contrast, the proposed LLM-Advisor (with GPT-4o) improves cost efficiency for 72.37% of A*-planned paths, 69.47% of RRT*-planned paths, and 78.70% of LLM-A*-planned paths. On the MultiTerraPath dataset, LLM-Advisor demonstrates stronger performance on the hard subset, further validating its applicability to real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01267v2">Iterative In-Context Learning to Enhance LLMs Abstract Reasoning: The Case-Study of Algebraic Tasks</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted at KNLP 2026 - ACM SAC 2026 Special Track on Knowledge and Natural Language Processing. https://knlp-sac.github.io/2026/index.html
    </div>
    <details class="paper-abstract">
      LLMs face significant challenges in systematic generalization, particularly when dealing with reasoning tasks requiring compositional rules and handling out-of-distribution examples. To address these challenges, we introduce an in-context learning methodology that improves the generalization capabilities of general purpose LLMs. Our approach employs an iterative example selection strategy, which incrementally constructs a tailored set of few-shot examples optimized to enhance model's performance on a given task. As a proof of concept, we apply this methodology to the resolution of algebraic expressions involving non-standard simplification rules, according to which the priority of addition and multiplication is changed. Our findings indicate that LLMs exhibit limited proficiency in these mathematical tasks. We further demonstrate that LLMs reasoning benefits from our iterative shot selection prompting strategy integrated with explicit reasoning instructions. Crucially, our experiments reveal that some LLMs achieve better generalization performances when prompted with simpler few-shot examples rather than complex ones following the test data distribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09536v1">Dynamic Multimodal Expression Generation for LLM-Driven Pedagogical Agents: From User Experience Perspective</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      In virtual reality (VR) educational scenarios, Pedagogical agents (PAs) enhance immersive learning through realistic appearances and interactive behaviors. However, most existing PAs rely on static speech and simple gestures. This limitation reduces their ability to dynamically adapt to the semantic context of instructional content. As a result, interactions often lack naturalness and effectiveness in the teaching process. To address this challenge, this study proposes a large language model (LLM)-driven multimodal expression generation method that constructs semantically sensitive prompts to generate coordinated speech and gesture instructions, enabling dynamic alignment between instructional semantics and multimodal expressive behaviors. A VR-based PA prototype was developed and evaluated through user experience-oriented subjective experiments. Results indicate that dynamically generated multimodal expressions significantly enhance learners' perceived learning effectiveness, engagement, and intention to use, while effectively alleviating feelings of fatigue and boredom during the learning process. Furthermore, the combined dynamic expression of speech and gestures notably enhances learners' perceptions of human-likeness and social presence. The findings provide new insights and design guidelines for building more immersive and naturally expressive intelligent PAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09533v1">Enhancing Debunking Effectiveness through LLM-based Personality Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 In: Computational Intelligence. IJCCI 2025. Springer, Cham (2026)
    </div>
    <details class="paper-abstract">
      This study proposes a novel methodology for generating personalized fake news debunking messages by prompting Large Language Models (LLMs) with persona-based inputs aligned to the Big Five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness. Our approach guides LLMs to transform generic debunking content into personalized versions tailored to specific personality profiles. To assess the effectiveness of these transformations, we employ a separate LLM as an automated evaluator simulating corresponding personality traits, thereby eliminating the need for costly human evaluation panels. Our results show that personalized messages are generally seen as more persuasive than generic ones. We also find that traits like Openness tend to increase persuadability, while Neuroticism can lower it. Differences between LLM evaluators suggest that using multiple models provides a clearer picture. Overall, this work demonstrates a practical way to create more targeted debunking messages exploiting LLMs, while also raising important ethical questions about how such technology might be used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09434v1">Common Sense vs. Morality: The Curious Case of Narrative Focus Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted at LREC 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across diverse real-world applications and user communities. As such, it is crucial that these models remain both morally grounded and knowledge-aware. In this work, we uncover a critical limitation of current LLMs -- their tendency to prioritize moral reasoning over commonsense understanding. To investigate this phenomenon, we introduce CoMoral, a novel benchmark dataset containing commonsense contradictions embedded within moral dilemmas. Through extensive evaluation of ten LLMs across different model sizes, we find that existing models consistently struggle to identify such contradictions without prior signal. Furthermore, we observe a pervasive narrative focus bias, wherein LLMs more readily detect commonsense contradictions when they are attributed to a secondary character rather than the primary (narrator) character. Our comprehensive analysis underscores the need for enhanced reasoning-aware training to improve the commonsense robustness of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08317v2">Automatic Paper Reviewing with Heterogeneous Graph Reasoning over LLM-Simulated Reviewer-Author Debates</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Existing paper review methods often rely on superficial manuscript features or directly on large language models (LLMs), which are prone to hallucinations, biased scoring, and limited reasoning capabilities. Moreover, these methods often fail to capture the complex argumentative reasoning and negotiation dynamics inherent in reviewer-author interactions. To address these limitations, we propose ReViewGraph (Reviewer-Author Debates Graph Reasoner), a novel framework that performs heterogeneous graph reasoning over LLM-simulated multi-round reviewer-author debates. In our approach, reviewer-author exchanges are simulated through LLM-based multi-agent collaboration. Diverse opinion relations (e.g., acceptance, rejection, clarification, and compromise) are then explicitly extracted and encoded as typed edges within a heterogeneous interaction graph. By applying graph neural networks to reason over these structured debate graphs, ReViewGraph captures fine-grained argumentative dynamics and enables more informed review decisions. Extensive experiments on three datasets demonstrate that ReViewGraph outperforms strong baselines with an average relative improvement of 15.73%, underscoring the value of modeling detailed reviewer-author debate structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08163v2">Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 26 pages, 6 figures, 4 tables; minor update, no content changes
    </div>
    <details class="paper-abstract">
      Recently, there has been increased interest in globally distributed training, which has the promise to both reduce training costs and democratize participation in building large-scale foundation models. However, existing models trained in a globally distributed manner are relatively small in scale and have only been trained with whitelisted participants. Therefore, they do not yet realize the full promise of democratized participation. In this report, we describe Covenant-72B, an LLM produced by the largest collaborative globally distributed pre-training run (in terms of both compute and model scale), which simultaneously allowed open, permissionless participation supported by a live blockchain protocol. We utilized a state-of-the-art communication-efficient optimizer, SparseLoCo, supporting dynamic participation with peers joining and leaving freely. Our model, pre-trained on approximately 1.1T tokens, performs competitively with fully centralized models pre-trained on similar or higher compute budgets, demonstrating that fully democratized, non-whitelisted participation is not only feasible, but can be achieved at unprecedented scale for a globally distributed pre-training run.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09403v1">LLM as a Meta-Judge: Synthetic Data for NLP Evaluation Metric Validation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 16 pages, 1 figure, 14 tables
    </div>
    <details class="paper-abstract">
      Validating evaluation metrics for NLG typically relies on expensive and time-consuming human annotations, which predominantly exist only for English datasets. We propose \textit{LLM as a Meta-Judge}, a scalable framework that utilizes LLMs to generate synthetic evaluation datasets via controlled semantic degradation of real data, replacing human judgment. We validate our approach using \textit{meta-correlation}, measuring the alignment between metric rankings derived from synthetic data and those from standard human benchmarks. Experiments across Machine Translation, Question Answering, and Summarization demonstrate that synthetic validation serves as a reliable proxy for human judgment, achieving meta-correlations exceeding 0.9 in multilingual QA and proves to be a viable alternative where human judgments are unavailable or too expensive to obtain. Our code and data will become publicly available upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16368v4">SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Camera-ready version for Neural Information Processing Systems (NeurIPS) 2025, Spotlight Paper
    </div>
    <details class="paper-abstract">
      How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard. To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions. We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08837v3">Debiasing International Attitudes: LLM Agents for Simulating US-China Perception Changes</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Submitted to TCSS
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative opportunities to address the longstanding challenge of modeling opinion evolution in computational social science. This study investigates how media influences cross-border attitudes - a key driver of global polarization - by developing an LLM-agent framework to disentangle sources of bias and assess LLMs' capacity for human-like opinion formation in response to external information. We introduce an LLM-agent-based framework that models U.S. citizens' attitudes toward China from 2005 to 2025. Our approach integrates large-scale news data with social media profiles to initialize agent populations, which then undergo cognitive-aware reflection and opinion updating. We propose three debiasing mechanisms: (1) fact elicitation, extracting neutral events from subjectively framed news; (2) a devil's advocate agent that simulates critical contextualization; and (3) counterfactual exposure to surface inherent model biases. Simulations with two state-of-the-art LLMs (Qwen3-14b and GPT4o) reveal the expected negative attitudinal trend following media exposure. While all three mechanisms mitigate this trend to varying degrees, results indicate that subjective news framing contributes only modestly to negative attitudes, whereas the devil's advocate agent proves most effective overall, suggesting that intermediate analytical steps can produce more human-like agent opinions. Notably, the counterfactual study reveals contradictory findings across models, suggesting region-specific inherent biases tied to models' geographic origins. By advancing understanding of LLM-based opinion formation and debiasing methods, this study contributes to developing more objective models that better align with human cognitive tendencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09337v1">Beyond Scaling: Assessing Strategic Reasoning and Rapid Decision-Making Capability of LLMs in Zero-sum Environments</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Code available
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved strong performance on static reasoning benchmarks, yet their effectiveness as interactive agents operating in adversarial, time-sensitive environments remains poorly understood. Existing evaluations largely treat reasoning as a single-shot capability, overlooking the challenges of opponent-aware decision-making, temporal constraints, and execution under pressure. This paper introduces Strategic Tactical Agent Reasoning (STAR) Benchmark, a multi-agent evaluation framework that assesses LLMs through 1v1 zero-sum competitive interactions, framing reasoning as an iterative, adaptive decision-making process. STAR supports both turn-based and real-time settings, enabling controlled analysis of long-horizon strategic planning and fast-paced tactical execution within a unified environment. Built on a modular architecture with a standardized API and fully implemented execution engine, STAR facilitates reproducible evaluation and flexible task customization. To move beyond binary win-loss outcomes, we introduce a Strategic Evaluation Suite that assesses not only competitive success but also the quality of strategic behavior, such as execution efficiency and outcome stability. Extensive pairwise evaluations reveal a pronounced strategy-execution gap: while reasoning-intensive models dominate turn-based settings, their inference latency often leads to inferior performance in real-time scenarios, where faster instruction-tuned models prevail. These results show that strategic intelligence in interactive environments depends not only on reasoning depth, but also on the ability to translate plans into timely actions, positioning STAR as a principled benchmark for studying this trade-off in competitive, dynamic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17385v4">Reinforcing Numerical Reasoning in LLMs for Tabular Prediction via Structural Priors</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Tabular prediction traditionally relies on gradient-boosted decision trees and deep learning models, which excel in specific tasks but lack interpretability and transferability. Reasoning large language models (LLMs) promise cross-task adaptability with transparent reasoning traces, yet their potential for tabular data remains unrealized. To bridge this gap, we propose a reasoning framework centered on Permutation Relative Policy Optimization (PRPO), a reinforcement learning method that encodes column-permutation invariance as a structural prior. By estimating advantages across label-preserving permutations, PRPO transforms sparse rewards into dense signals, activating latent numerical reasoning capabilities of LLMs with limited supervision. Extensive experiments show that our method matches fully supervised baselines and dominates in zero-shot settings, performing on par with 32-shot strong baselines. Remarkably, our 8B model significantly outperforms much larger LLMs, achieving up to a 53.17% improvement over DeepSeek-R1 (685B).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09309v1">Rescaling Confidence: What Scale Design Reveals About LLM Metacognition</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Verbalized confidence, in which LLMs report a numerical certainty score, is widely used to estimate uncertainty in black-box settings, yet the confidence scale itself (typically 0--100) is rarely examined. We show that this design choice is not neutral. Across six LLMs and three datasets, verbalized confidence is heavily discretized, with more than 78% of responses concentrating on just three round-number values. To investigate this phenomenon, we systematically manipulate confidence scales along three dimensions: granularity, boundary placement, and range regularity, and evaluate metacognitive sensitivity using meta-d'. We find that a 0--20 scale consistently improves metacognitive efficiency over the standard 0--100 format, while boundary compression degrades performance and round-number preferences persist even under irregular ranges. These results demonstrate that confidence scale design directly affects the quality of verbalized uncertainty and should be treated as a first-class experimental variable in LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09297v1">TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) has exhibited strong reasoning ability in text-based contexts across various domains, yet the limitation of context window poses challenges for the model on long-range inference tasks and necessitates a memory storage system. While many current storage approaches have been proposed with episodic notes and graph representations of memory, retrieval methods still primarily rely on predefined workflows or static similarity top-k over embeddings. To address this inflexibility, we introduced a novel tool-augmented autonomous memory retrieval framework (TA-Mem), which contains: (1) a memory extraction LLM agent which is prompted to adaptively chuck an input into sub-context based on semantic correlation, and extract information into structured notes, (2) a multi-indexed memory database designed for different types of query methods including both key-based lookup and similarity-based retrieval, (3) a tool-augmented memory retrieval agent which explores the memory autonomously by selecting appropriate tools provided by the database based on the user input, and decides whether to proceed to the next iteration or finalizing the response after reasoning on the fetched memories. The TA-Mem is evaluated on the LoCoMo dataset, achieving significant performance improvements over existing baseline approaches. In addition, an analysis of tool use across different question types also demonstrates the adaptivity of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09250v1">Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Personalized large language models (LLMs) rely on memory retrieval to incorporate user-specific histories, preferences, and contexts. Existing approaches either overload the LLM by feeding all the user's past memory into the prompt, which is costly and unscalable, or simplify retrieval into a one-shot similarity search, which captures only surface matches. Cognitive science, however, shows that human memory operates through a dual process: Familiarity, offering fast but coarse recognition, and Recollection, enabling deliberate, chain-like reconstruction for deeply recovering episodic content. Current systems lack both the ability to perform recollection retrieval and mechanisms to adaptively switch between the dual retrieval paths, leading to either insufficient recall or the inclusion of noise. To address this, we propose RF-Mem (Recollection-Familiarity Memory Retrieval), a familiarity uncertainty-guided dual-path memory retriever. RF-Mem measures the familiarity signal through the mean score and entropy. High familiarity leads to the direct top-K Familiarity retrieval path, while low familiarity activates the Recollection path. In the Recollection path, the system clusters candidate memories and applies alpha-mix with the query to iteratively expand evidence in embedding space, simulating deliberate contextual reconstruction. This design embeds human-like dual-process recognition into the retriever, avoiding full-context overhead and enabling scalable, adaptive personalization. Experiments across three benchmarks and corpus scales demonstrate that RF-Mem consistently outperforms both one-shot retrieval and full-context reasoning under fixed budget and latency constraints. Our code can be found in the Reproducibility Statement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11067v1">Summarize Before You Speak with ARACH: A Training-Free Inference-Time Plug-In for Enhancing LLMs via Global Attention Reallocation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve remarkable performance, yet further gains often require costly training. This has motivated growing interest in post-training techniques-especially training-free approaches that improve models at inference time without updating weights. Most training-free methods treat the model as a black box and improve outputs via input/output-level interventions, such as prompt design and test-time scaling through repeated sampling, reranking/verification, or search. In contrast, they rarely offer a plug-and-play mechanism to intervene in a model's internal computation. We propose ARACH(Attention Reallocation via an Adaptive Context Hub), a training-free inference-time plug-in that augments LLMs with an adaptive context hub to aggregate context and reallocate attention. Extensive experiments across multiple language modeling tasks show consistent improvements with modest inference overhead and no parameter updates. Attention analyses further suggest that ARACH mitigates the attention sink phenomenon. These results indicate that engineering a model's internal computation offers a distinct inference-time strategy, fundamentally different from both prompt-based test-time methods and training-based post-training approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09231v1">Cognitively Layered Data Synthesis for Domain Adaptation of LLMs to Space Situational Awareness</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate exceptional performance on general-purpose tasks. however, transferring them to complex engineering domains such as space situational awareness (SSA) remains challenging owing to insufficient structural alignment with mission chains, the absence of higher-order cognitive supervision, and poor correspondence between data quality criteria and engineering specifications. The core bottleneck is the construction of high-quality supervised fine-tuning (SFT) datasets. To this end, we propose BD-FDG (Bloom's Taxonomy-based Domain-specific Fine-tuning Data Generation), a framework that addresses incomplete knowledge coverage, shallow cognitive depth, and limited quality controllability through three mechanisms: structured knowledge organization, cognitively layered question modeling, and automated quality control. The framework uses a knowledge tree to ensure structured corpus coverage, designs a question generation scheme spanning nine categories and six cognitive levels from Remember to Create to produce samples with a continuous difficulty gradient, and applies a multidimensional scoring pipeline to enforce domain rigor and consistency. Using BD-FDG, we construct SSA-SFT, a domain dataset of approximately 230K samples, and fine-tune Qwen3-8B to obtain SSA-LLM-8B. Experiments show that SSA-LLM-8B achieves relative BLEU-1 improvements of 144\% (no-think) and 176\% (think) on the domain test set and a win rate of 82.21\% over the baseline in arena comparisons, while largely preserving general benchmark performance (MMLU-Pro, MATH-500). These results validate SFT data construction driven by cognitive layering as an effective paradigm for complex engineering domains and provide a transferable framework for domain-specific LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09216v1">PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 13 pages, 13 figures
    </div>
    <details class="paper-abstract">
      On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-memory (PIM) in both academia and industry. However, practical PIM-enabled systems face two obstacles between these phases, a memory attribute inconsistency in which prefill favors placing weights in a cacheable region for reuse whereas decode requires weights in a non-cacheable region to reliably trigger PIM, and a weight layout inconsistency between host-friendly and PIM-aware layouts. To address these problems, we introduce \textit{PIM-SHERPA}, a software-only method for efficient on-device LLM inference by resolving PIM memory attribute and layout inconsistencies. PIM-SHERPA provides two approaches, DRAM double buffering (DDB), which keeps a single PIM-aware weights in the non-cacheable region while prefetching the swizzled weights of the next layer into small cacheable buffers, and online weight rearrangement with swizzled memory copy (OWR), which performs the on-demand swizzled memory copy immediately before GEMM. Compared to a baseline PIM emulation system, PIM-SHERPA achieves approximately 47.8 - 49.7\% memory capacity savings while maintaining comparable performance to the theoretical maximum on the Llama 3.2 model. To the best of our knowledge, this is the first work to identify the memory attribute inconsistency and propose effective solutions on product-level PIM-enabled systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10072v1">Why LLMs Fail: A Failure Analysis and Partial Success Measurement for Automated Security Patch Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise for Automated Program Repair (APR), yet their effectiveness on security vulnerabilities remains poorly characterized. This study analyzes 319 LLM-generated security patchesacross 64 Java vulnerabilities from the Vul4J benchmark. Using tri-axis evaluation (compilation, security via PoV tests, functionality via test suites), the analysis reveals that only 24.8% of patches achieve full correctness, while 51.4% fail both security and functionality. The dominant failure mode is semantic misunderstanding: LLMs produce syntactically valid code but apply incorrect repair strategies. The proposed Security Repair Score (SRS) quantifies this gap, showing LLMs preserve functionality (mean 0.832) but struggle with security (mean 0.251). Vulnerability type strongly predicts difficulty, with fix rates ranging from 0% (input validation) to 45% (infinite loop). These findings demonstrate that LLM security patches require rigorous validation before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01290v2">Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      High-quality time series (TS) data are essential for ensuring TS model performance, rendering research on rating TS data quality indispensable. Existing methods have shown promising rating accuracy within individual domains, primarily by extending data quality rating techniques such as influence functions and Shapley values to account for temporal characteristics. However, they neglect the fact that real-world TS data can span vastly different domains and exhibit distinct properties, hampering the accurate and efficient rating of diverse TS data. In this paper, we propose TSRating, a novel and unified framework for rating the quality of time series data crawled from diverse domains. TSRating leverages LLMs' inherent ample knowledge, acquired during their extensive pretraining, to comprehend and discern quality differences in diverse TS data. We verify this by devising a series of prompts to elicit quality comparisons from LLMs for pairs of TS samples. We then fit a dedicated rating model, termed TSRater, to convert the LLMs' judgments into efficient quality predictions by inferring future TS samples through TSRater's inference. To ensure cross-domain adaptability, we develop a meta-learning scheme to train TSRater on quality comparisons collected from nine distinct domains. To improve training efficiency, we employ signSGD for inner-loop updates, thus circumventing the demanding computation of hypergradients. Extensive experimental results on eleven benchmark datasets across three time series tasks, each using both conventional TS models and TS foundation models, demonstrate that TSRating outperforms baselines in terms of estimation accuracy, efficiency, and domain adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04818v2">LLM-Grounded Explainable AI for Supply Chain Risk Early Warning via Temporal Graph Attention Networks</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Disruptions at critical logistics nodes pose severe risks to global supply chains, yet existing risk prediction systems typically prioritize forecasting accuracy without providing operationally interpretable early warnings. This paper proposes an evidence-grounded framework that jointly performs supply chain bottleneck prediction and faithful natural-language risk explanation by coupling a Temporal Graph Attention Network (TGAT) with a structured large language model (LLM) reasoning module. Using maritime hubs as a representative case study for global supply chain nodes, daily spatial graphs are constructed from Automatic Identification System (AIS) broadcasts, where inter-node interactions are modeled through attention-based message passing. The TGAT predictor captures spatiotemporal risk dynamics, while model-internal evidence -- including feature z-scores and attention-derived neighbor influence -- is transformed into structured prompts that constrain LLM reasoning to verifiable model outputs. To evaluate explanatory reliability, we introduce a directional-consistency validation protocol that quantitatively measures agreement between generated risk narratives and underlying statistical evidence. Experiments on six months of real-world logistics data demonstrate that the proposed framework outperforms baseline models, achieving a test AUC of 0.761, AP of 0.344, and recall of 0.504 under a strict chronological split while producing early warning explanations with 99.6\% directional consistency. Results show that grounding LLM generation in graph-model evidence enables interpretable and auditable risk reporting without sacrificing predictive performance. The framework provides a practical pathway toward operationally deployable explainable AI for supply chain risk early warning and resilience management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09181v1">Evaluating the Practical Effectiveness of LLM-Driven Index Tuning with Microsoft Database Tuning Advisor</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Index tuning is critical for the performance of modern database systems. Industrial index tuners, such as the Database Tuning Advisor (DTA) developed for Microsoft SQL Server, rely on the "what-if" API provided by the query optimizer to estimate the cost of a query given an index configuration, which can lead to suboptimal recommendations when the estimations are inaccurate. Large language model (LLM) offers a new approach to index tuning, with knowledge learned from web-scale training datasets. However, the effectiveness of LLM-driven index tuning, especially beyond what is already achieved by commercial index tuners, remains unclear. In this paper, we study the practical effectiveness of LLM-driven index tuning using both industrial benchmarks and real-world enterprise customer workloads, and compare it with DTA. Our results show that although DTA is generally more reliable, with a few invocations, LLM can identify configurations that significantly outperform those found by DTA in execution time in a considerable number of cases, highlighting its potential as a complementary technique. We also observe that LLM's reasoning captures human-intuitive insights that may be distilled to potentially improve DTA. However, adopting LLM-driven index tuning in production remains challenging due to its substantial performance variance, limited and often negative impact when directly integrated into DTA, and the high cost of performance validation. This work provides motivation, lessons, and practical insights that will inspire future work on LLM-driven index tuning both in academia and industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09180v1">DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Spoken dialog systems with cascaded ASR-LLM-TTS modules retain strong LLM intelligence, but VAD segmentation often forces half-duplex turns and brittle control. On the other hand, VAD-free end-to-end model support full-duplex interaction but is hard to maintain conversational intelligence. In this paper, we present DuplexCascade, a VAD-free cascaded streaming pipeline for full-duplex speech-to-speech dialogue. Our key idea is to convert conventional utterance-wise long turns into chunk-wise micro-turn interactions, enabling rapid bidirectional exchange while preserving the strengths of a capable text LLM. To reliably coordinate turn-taking and response timing, we introduce a set of conversational special control tokens that steer the LLM's behavior under streaming constraints. On Full-DuplexBench and VoiceBench, DuplexCascade delivers state-of-the-art full-duplex turn-taking and strong conversational intelligence among open-source speech-to-speech dialogue systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18315v2">CovertComBench: A First Domain-Specific Testbed for LLMs in Wireless Covert Communication</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 6 pages, corrected a typo: "DeepSeek-R1:70B" was previously incorrectly written as "DeepSeek-V3.2:70B"
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into wireless networks presents significant potential for automating system design. However, unlike conventional throughput maximization, Covert Communication (CC) requires optimizing transmission utility under strict detection-theoretic constraints, such as Kullback-Leibler divergence limits. Existing benchmarks primarily focus on general reasoning or standard communication tasks and do not adequately evaluate the ability of LLMs to satisfy these rigorous security constraints. To address this limitation, we introduce CovertComBench, a unified benchmark designed to assess LLM capabilities across the CC pipeline, encompassing conceptual understanding (MCQs), optimization derivation (ODQs), and code generation (CGQs). Furthermore, we analyze the reliability of automated scoring within a detection-theoretic ``LLM-as-Judge'' framework. Extensive evaluations across state-of-the-art models reveal a significant performance discrepancy. While LLMs achieve high accuracy in conceptual identification (81%) and code implementation (83%), their performance in the higher-order mathematical derivations necessary for security guarantees ranges between 18% and 55%. This limitation indicates that current LLMs serve better as implementation assistants rather than autonomous solvers for security-constrained optimization. These findings suggest that future research should focus on external tool augmentation to build trustworthy wireless AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09155v2">OPENXRD: A Comprehensive Benchmark Framework for LLM/MLLM XRD Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted at Digital Discovery (Royal Society of Chemistry)
    </div>
    <details class="paper-abstract">
      We introduce OPENXRD, a comprehensive benchmarking framework for evaluating large language models (LLMs) and multimodal LLMs (MLLMs) in crystallography question answering. The framework measures context assimilation, or how models use fixed, domain-specific supporting information during inference. The framework includes 217 expert-curated X-ray diffraction (XRD) questions covering fundamental to advanced crystallographic concepts, each evaluated under closed-book (without context) and open-book (with context) conditions, where the latter includes concise reference passages generated by GPT-4.5 and refined by crystallography experts. We benchmark 74 state-of-the-art LLMs and MLLMs, including GPT-4, GPT-5, O-series, LLaVA, LLaMA, QWEN, Mistral, and Gemini families, to quantify how different architectures and scales assimilate external knowledge. Results show that mid-sized models (7B--70B parameters) gain the most from contextual materials, while very large models often show saturation or interference and the largest relative gains appear in small and mid-sized models. Expert-reviewed materials provide significantly higher improvements than AI-generated ones even when token counts are matched, confirming that content quality, not quantity, drives performance. OPENXRD offers a reproducible diagnostic benchmark for assessing reasoning, knowledge integration, and guidance sensitivity in scientific domains, and provides a foundation for future multimodal and retrieval-augmented crystallography systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09161v1">Wrong Code, Right Structure: Learning Netlist Representations from Imperfect LLM-Generated RTL</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Learning effective netlist representations is fundamentally constrained by the scarcity of labeled datasets, as real designs are protected by Intellectual Property (IP) and costly to annotate. Existing work therefore focuses on small-scale circuits with clean labels, limiting scalability to realistic designs. Meanwhile, Large Language Models (LLMs) can generate Register-Transfer-Level (RTL) at scale, but their functional incorrectness has hindered their use in circuit analysis. In this work, we make a key observation: even when LLM-Generated RTL is functionally imperfect, the synthesized netlists still preserve structural patterns that are strongly indicative of the intended functionality. Building on this insight, we propose a cost-effective data augmentation and training framework that systematically exploits imperfect LLM-Generated RTL as training data for netlist representation learning, forming an end-to-end pipeline from automated code generation to downstream tasks. We conduct evaluations on circuit functional understanding tasks, including sub-circuit boundary identification and component classification, across benchmarks of increasing scales, extending the task scope from operator-level to IP-level. The evaluations demonstrate that models trained on our noisy synthetic corpus generalize well to real-world netlists, matching or even surpassing methods trained on scarce high-quality data and effectively breaking the data bottleneck in circuit representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09154v1">Bioalignment: Measuring and Improving LLM Disposition Toward Biological Systems for AI Safety</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained on internet-scale corpora can exhibit systematic biases that increase the probability of unwanted behavior. In this study, we examined potential biases towards synthetic vs. biological technological solutions across four domains (materials, energy, manufacturing, and algorithms). A sample of 5 frontier and 5 open-weight models were measured using 50 curated Bioalignment prompts with a Kelly criterion-inspired evaluation framework. According to this metric, most models were not bioaligned in that they exhibit biases in favor of synthetic (non-biological) solutions. We next examined if fine-tuning could increase the preferences of two open-weight models, Llama 3.2-3B-Instruct and Qwen2.5-3B-Instruct, for biological-based approaches. A curated corpus of ~22M tokens from 6,636 PMC articles emphasizing biological problem-solving was used first to fine-tune Llama 3B with a mixed corpus of continued training and instruction-formatted. This was then extended to Qwen 3B using instruction-formatted only. We found that QLoRA fine-tuning significantly increased the scoring of biological solutions for both models without degrading general capabilities (Holm-Bonferroni-corrected p < 0.001 and p < 0.01, respectively). This suggests that even a small amount of fine-tuning can change how models weigh the relative value of biological and bioinspired vs. synthetic approaches. Although this work focused on small open-weight LLMs, it may be extensible to much larger models and could be used to develop models that favor bio-based approaches. We release the benchmark, corpus, code, and adapter weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15030v4">A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge devices enhances privacy but faces performance hurdles due to limited resources. We introduce a systematic methodology to evaluate on-device LLMs, balancing capability, efficiency, and resource constraints. Through an extensive analysis of models (0.5B-14B) and seven post-training quantization (PTQ) methods on commodity hardware, we demonstrate that: 1) Heavily quantized large models consistently outperform smaller, high-precision models, with a performance threshold at ~3.5 effective bits-per-weight (BPW); 2) Resource utilization scales linearly with BPW, though power and memory footprints vary by quantization algorithm; and 3) With a reduction in model size, the primary constraint on throughput transitions from communication overhead to computational latency. We conclude by offering guidelines for optimizing LLMs in resource-constrained edge environments. Our codebase is available at https://anonymous.4open.science/r/LLMOnDevice/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.09916v2">"Should I Give Up Now?" Investigating LLM Pitfalls in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Software engineers are increasingly incorporating AI assistants into their workflows to enhance productivity and alleviate cognitive load. However, experiences with large language models (LLMs) such as ChatGPT vary widely. While some engineers find them useful, others deem them counterproductive due to inaccuracies in their responses. Researchers have also observed that ChatGPT often provides incorrect information. Given these limitations, it is crucial to determine how to effectively integrate LLMs into software engineering (SE) workflow. Analyzing data from 26 participants in a complex web development task, we identified nine failure types categorized into incorrect or incomplete responses, cognitive overload, and context loss. Users attempted to mitigate these issues through scaffolding, prompt clarification, and debugging. However, 17 participants ultimately chose to abandon ChatGPT due to persistent failures. Our quantitative analysis revealed that unhelpful responses increased the likelihood of abandonment by a factor of 11, while each additional prompt reduced abandonment probability by 17%. This study advances the understanding of human-AI interaction in SE tasks and outlines directions for future research and tooling support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09127v1">Chaotic Dynamics in Multi-LLM Deliberation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Main text: 6 pages, 4 figures; Supplementary Information: 14 pages, 7 supplementary figures
    </div>
    <details class="paper-abstract">
      Collective AI systems increasingly rely on multi-LLM deliberation, but their stability under repeated execution remains poorly characterized. We model five-agent LLM committees as random dynamical systems and quantify inter-run sensitivity using an empirical Lyapunov exponent ($\hatλ$) derived from trajectory divergence in committee mean preferences. Across 12 policy scenarios, a factorial design at $T=0$ identifies two independent routes to instability: role differentiation in homogeneous committees and model heterogeneity in no-role committees. Critically, these effects appear even in the $T=0$ regime where practitioners often expect deterministic behavior. In the HL-01 benchmark, both routes produce elevated divergence ($\hatλ=0.0541$ and $0.0947$, respectively), while homogeneous no-role committees also remain in a positive-divergence regime ($\hatλ=0.0221$). The combined mixed+roles condition is less unstable than mixed+no-role ($\hatλ=0.0519$ vs $0.0947$), showing non-additive interaction. Mechanistically, Chair-role ablation reduces $\hatλ$ most strongly, and targeted protocol variants that shorten memory windows further attenuate divergence. These results support stability auditing as a core design requirement for multi-LLM governance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11066v1">Exploring Collatz Dynamics with Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We investigate structural properties of the Collatz iteration through two phenomena observed in large computational exploration: modular scrambling of residue classes and a burst--gap decomposition of trajectories. We prove several structural results, including a modular scrambling lemma showing that the gap-return map acts as an exact bijection on high bits, a persistent exit lemma characterizing gap structure after persistent states, and a decay property for known portions of binary representations under gap-return dynamics. We further prove that, in the modular model, gap lengths and $2$-adic valuations follow geometric distributions, while persistent run lengths are geometric with expected burst length $E[B]=2$; together these predict strict orbit contraction. These results suggest a conditional framework in which convergence would follow from suitable orbitwise hypotheses on burst and gap lengths, which in turn are suggested by an orbit equidistribution conjecture. However, the key hypotheses remain open, and the framework is exploratory rather than a complete reduction. The paper also documents the human-LLM collaboration through which these observations were developed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09085v1">Not All News Is Equal: Topic- and Event-Conditional Sentiment from Finetuned LLMs for Aluminum Price Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      By capturing the prevailing sentiment and market mood, textual data has become increasingly vital for forecasting commodity prices, particularly in metal markets. However, the effectiveness of lightweight, finetuned large language models (LLMs) in extracting predictive signals for aluminum prices, and the specific market conditions under which these signals are most informative, remains under-explored. This study generates monthly sentiment scores from English and Chinese news headlines (Reuters, Dow Jones Newswires, and China News Service) and integrates them with traditional tabular data, including base metal indices, exchange rates, inflation rates, and energy prices. We evaluate the predictive performance and economic utility of these models through long-short simulations on the Shanghai Metal Exchange from 2007 to 2024. Our results demonstrate that during periods of high volatility, Long Short-Term Memory (LSTM) models incorporating sentiment data from a finetuned Qwen3 model (Sharpe ratio 1.04) significantly outperform baseline models using tabular data alone (Sharpe ratio 0.23). Subsequent analysis elucidates the nuanced roles of news sources, topics, and event types in aluminum price forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09065v1">Learning Adaptive LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Decoding from large language models (LLMs) typically relies on fixed sampling hyperparameters (e.g., temperature, top-p), despite substantial variation in task difficulty and uncertainty across prompts and individual decoding steps. We propose to learn adaptive decoding policies that dynamically select sampling strategies at inference time, conditioned on available compute resources. Rather than fine-tuning the language model itself, we introduce lightweight decoding adapters trained with reinforcement learning and verifiable terminal rewards (e.g. correctness on math and coding tasks). At the sequence level, we frame decoding as a contextual bandit problem: a policy selects a decoding strategy (e.g. greedy, top-k, min-p) for each prompt, conditioned on the prompt embedding and a parallel sampling budget. At the token level, we model decoding as a partially observable Markov decision process (POMDP), where a policy selects sampling actions at each token step based on internal model features and the remaining token budget. Experiments on the MATH and CodeContests benchmarks show that the learned adapters improve the accuracy-budget tradeoff: on MATH, the token-level adapter improves Pass@1 accuracy by up to 10.2% over the best static baseline under a fixed token budget, while the sequence-level adapter yields 2-3% gains under fixed parallel sampling. Ablation analyses support the contribution of both sequence- and token-level adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09046v1">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have witnessed explosive growth, offering higher privacy and availability compared to cloud-side LLMs. During LLM inference, both model weights and user data are valuable, and attackers may even compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead due to its inflexible isolation of memory and the NPU. To address these challenges, this paper introduces FlexServe, a fast and secure LLM serving system for mobile devices. It first introduces a Flexible Resource Isolation mechanism to construct Flexible Secure Memory (Flex-Mem) and Flexible Secure NPU (Flex-NPU). Both memory pages and the NPU can be efficiently switched between unprotected and protected modes. Based on these mechanisms, FlexServe designs a fast and secure LLM inference framework within TrustZone's secure world. The LLM-Aware Memory Management and Secure Inference Pipeline are introduced to accelerate inference. A Multi-Model Scheduler is proposed to optimize multi-model workflows. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves an average $10.05\times$ speedup in Time to First Token (TTFT) compared to the strawman, and an average $2.44\times$ TTFT speedup compared to an optimized strawman with pipeline and secure NPU enabled. For multi-model agent workflows, the end-to-end speedup is up to $24.30\times$ and $4.05\times$ compared to the strawman and optimized strawman, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09044v1">Synergistic Directed Execution and LLM-Driven Analysis for Zero-Day AI-Generated Malware Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 18 pages, CRIS 2026
    </div>
    <details class="paper-abstract">
      The weaponization of LLMs for automated malware generation poses an existential threat to conventional detection paradigms. AI-generated malware exhibits polymorphic, metamorphic, and context-aware evasion capabilities that render signature-based and shallow heuristic defenses obsolete. This paper introduces a novel hybrid analysis framework that synergistically combines \emph{concolic execution} with \emph{LLM-augmented path prioritization} and \emph{deep-learning-based vulnerability classification} to detect zero-day AI-generated malware with provable guarantees. We formalize the detection problem within a first-order temporal logic over program execution traces, define a lattice-theoretic abstraction for path constraint spaces, and prove both the \emph{soundness} and \emph{relative completeness} of our detection algorithm, assuming classifier correctness. The framework introduces three novel algorithms: (i) an LLM-guided concolic exploration strategy that reduces the average number of explored paths by 73.2\% compared to depth-first search while maintaining equivalent malicious-path coverage; (ii) a transformer-based path-constraint classifier trained on symbolic execution traces; and (iii) a feedback loop that iteratively refines the LLM's prioritization policy using reinforcement learning from detection outcomes. We provide a comprehensive implementation built upon \texttt{angr} 9.2, \texttt{Z3} 4.12, Hugging Face Transformers 4.38, and PyTorch 2.2, with configuration details enabling reproducibility. Experimental evaluation on the EMBER, Malimg, SOREL-20M, and a novel AI-Gen-Malware benchmark comprising 2{,}500 LLM-synthesized samples demonstrates that achieves 98.7\% accuracy on conventional malware and 97.5\% accuracy on AI-generated threats, outperforming ClamAV, YARA, MalConv, and EMBER-GBDT baselines by margins of 8.4--52.2 percentage points on AI-generated samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09036v1">SCALAR: Learning and Composing Skills through LLM Guided Symbolic Planning and Deep RL Grounding</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Best Paper Award Honorable Mention at NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning
    </div>
    <details class="paper-abstract">
      LM-based agents excel when given high-level action APIs but struggle to ground language into low-level control. Prior work has LLMs generate skills or reward functions for RL, but these one-shot approaches lack feedback to correct specification errors. We introduce SCALAR, a bidirectional framework coupling LLM planning with RL through a learned skill library. The LLM proposes skills with preconditions and effects; RL trains policies for each skill and feeds back execution results to iteratively refine specifications, improving robustness to initial errors. Pivotal Trajectory Analysis corrects LLM priors by analyzing RL trajectories; Frontier Checkpointing optionally saves environment states at skill boundaries to improve sample efficiency. On Craftax, SCALAR achieves 88.2% diamond collection, a 1.9x improvement over the best baseline, and reaches the Gnomish Mines 9.1% of the time where prior methods fail entirely.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09023v1">The Missing Memory Hierarchy: Demand Paging for LLM Context Windows</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      The context window of a large language model is not memory. It is L1 cache: a small, fast, expensive resource that the field treats as the entire memory system. There is no L2, no virtual memory, no paging. Every tool definition, every system prompt, and every stale tool result occupies context for the lifetime of the session. The result is measurable: across 857 production sessions and 4.45 million effective input tokens, 21.8% is structural waste. We present Pichay, a demand paging system for LLM context windows. Implemented as a transparent proxy between client and inference API, Pichay interposes on the message stream to evict stale content, detect page faults when the model re-requests evicted material, and pin working-set pages identified by fault history. In offline replay across 1.4 million simulated evictions, the fault rate is 0.0254%. In live production deployment over 681turns, the system reduces context consumption by up to 93% (5,038KB to 339KB); under extreme sustained pressure, the system remains operational but exhibits the expected thrashing pathology, with repeated fault-in of evicted content. The key observation is that the problems the field faces, such as context limits, attention degradation, cost scaling, lost state across sessions, are virtual memory problems wearing different clothes. The solutions exist: working set theory (Denning, 1968), demand paging, fault-driven replacement policies, and memory hierarchies with multiple eviction-managed levels. We describe the architecture of a full memory hierarchy for LLM systems (L1 through persistent storage), report on the first three levels deployed in production use (L1 eviction, L2 fault-driven pinning, L3 model-initiated conversation compaction), and identify cross-session memory as the remaining frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09022v1">MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Multi-turn, multi-agent LLM game evaluations often exhibit substantial run-to-run variance. In long-horizon interactions, small early deviations compound across turns and are amplified by multi-agent coupling. This biases win rate estimates and makes rankings unreliable across repeated tournaments. Prompt choice worsens this further by producing different effective policies. We address both instability and underperformance with MEMO (Memory-augmented MOdel context optimization), a self-play framework that optimizes inference-time context by coupling retention and exploration. Retention maintains a persistent memory bank that stores structured insights from self-play trajectories and injects them as priors during later play. Exploration runs tournament-style prompt evolution with uncertainty-aware selection via TrueSkill, and uses prioritized replay to revisit rare and decisive states. Across five text-based games, MEMO raises mean win rate from 25.1% to 49.5% for GPT-4o-mini and from 20.9% to 44.3% for Qwen-2.5-7B-Instruct, using $2,000$ self-play games per task. Run-to-run variance also drops, giving more stable rankings across prompt variations. These results suggest that multi-agent LLM game performance and robustness have substantial room for improvement through context optimization. MEMO achieves the largest gains in negotiation and imperfect-information games, while RL remains more effective in perfect-information settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09012v1">"Who wants to be nagged by AI?": Investigating the Effects of Agreeableness on Older Adults' Perception of LLM-Based Voice Assistants' Explanations</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 To be published as a poster extended abstract at CHI 2026
    </div>
    <details class="paper-abstract">
      LLM-based voice assistants (VAs) increasingly support older adults aging in place, yet how an assistant's agreeableness shapes explanation perception remains underexplored. We conducted a study(N=70) examining how VA agreeableness influences older adults' perceptions of explanations across routine and emergency home scenarios. High-agreeableness assistants were perceived as more trustworthy, empathetic, and likable, but these benefits diminished in emergencies where clarity outweighed warmth. Agreeableness did not affect perceived intelligence, suggesting social tone and competence are separable dimensions. Real-time environmental explanations outperformed history-based ones, and agreeable older adults penalized low-agreeableness assistants more strongly. These findings show the need to move beyond a one-size-fits-all approach to AI explainability, while balancing personality, context, and audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08999v1">Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating multiple reasoning trajectories, leading to substantial additional computational overhead. This paper introduces a confidence-aware decision framework that analyzes a single completed reasoning trajectory to adaptively select between single-path and multi-path reasoning. The framework is trained using sentence-level numeric and linguistic features extracted from intermediate reasoning states in the MedQA dataset and generalizes effectively to MathQA, MedMCQA, and MMLU without additional fine-tuning. Experimental results show that the proposed method maintains accuracy comparable to multi-path baselines while using up to 80\% fewer tokens. These findings demonstrate that reasoning trajectories contain rich signals for uncertainty estimation, enabling a simple, transferable mechanism to balance accuracy and efficiency in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08993v1">Arbiter: Detecting Interference in LLM Agent System Prompts</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      System prompts for LLM-based coding agents are software artifacts that govern agent behavior, yet lack the testing infrastructure applied to conventional software. We present Arbiter, a framework combining formal evaluation rules with multi-model LLM scouring to detect interference patterns in system prompts. Applied to three major coding agent system prompts: Claude Code (Anthropic), Codex CLI (OpenAI), and Gemini CLI (Google), we identify 152 findings across the undirected scouring phase and 21 hand-labeled interference patterns in directed analysis of one vendor. We show that prompt architecture (monolithic, flat, modular) strongly correlates with observed failure class but not with severity, and that multi-model evaluation discovers categorically different vulnerability classes than single-model analysis. One scourer finding was structural data loss in Gemini CLI's memory system was consistent with an issue filed and patched by Google, which addressed the symptom without addressing the schema-level root cause identified by the scourer. Total cost of cross-vendor analysis: \$0.27 USD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08954v1">A Consensus-Driven Multi-LLM Pipeline for Missing-Person Investigations</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted to CAC: Applied Computing & Automation Conferences 2026. 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The first 72 hours of a missing-person investigation are critical for successful recovery. Guardian is an end-to-end system designed to support missing-child investigation and early search planning. This paper presents the Guardian LLM Pipeline, a multi-model system in which LLMs are used for intelligent information extraction and processing related to missing-person search operations. The pipeline coordinates end-to-end execution across task-specialized LLM models and invokes a consensus LLM engine that compares multiple model outputs and resolves disagreements. The pipeline is further strengthened by QLoRA-based fine-tuning, using curated datasets. The presented design aligns with prior work on weak supervision and LLM-assisted annotation, emphasizing conservative, auditable use of LLMs as structured extractors and labelers rather than unconstrained end-to-end decision makers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08936v1">VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Speech Large Language Models (LLMs) show great promise for speech emotion recognition (SER) via generative interfaces. However, shifting from closed-set classification to open text generation introduces zero-shot stochasticity, making evaluation highly sensitive to prompts. Additionally, conventional speech LLMs benchmarks overlook the inherent ambiguity of human emotion. Hence, we present VoxEmo, a comprehensive SER benchmark encompassing 35 emotion corpora across 15 languages for Speech LLMs. VoxEmo provides a standardized toolkit featuring varying prompt complexities, from direct classification to paralinguistic reasoning. To reflect real-world perception/application, we introduce a distribution-aware soft-label protocol and a prompt-ensemble strategy that emulates annotator disagreement. Experiments reveal that while zero-shot speech LLMs trail supervised baselines in hard-label accuracy, they uniquely align with human subjective distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08933v1">Interpretable Markov-Based Spatiotemporal Risk Surfaces for Missing-Child Search Planning with Reinforcement Learning and LLM-Based Quality Assurance</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 14 pages, 7 figures. Accepted at ICEIS 2026 (International Conference on Enterprise Information Systems)
    </div>
    <details class="paper-abstract">
      The first 72 hours of a missing-child investigation are critical for successful recovery. However, law enforcement agencies often face fragmented, unstructured data and a lack of dynamic, geospatial predictive tools. Our system, Guardian, provides an end-to-end decision-support system for missing-child investigation and early search planning. It converts heterogeneous, unstructured case documents into a schema-aligned spatiotemporal representation, enriches cases with geocoding and transportation context, and provides probabilistic search products spanning 0-72 hours. In this paper, we present an overview of Guardian as well as a detailed description of a three-layer predictive component of the system. The first layer is a Markov chain, a sparse, interpretable model with transitions incorporating road accessibility costs, seclusion preferences, and corridor bias with separate day/night parameterizations. The Markov chain's output prediction distributions are then transformed into operationally useful search plans by the second layer's reinforcement learning. Finally, the third layer's LLM performs post hoc validation of layer 2 search plans prior to their release. Using a synthetic but realistic case study, we report quantitative outputs across 24/48/72-hour horizons and analyze sensitivity, failure modes, and tradeoffs. Results show that the proposed predictive system with the three-layer architecture produces interpretable priors for zone optimization and human review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08877v1">Quantifying the Accuracy and Cost Impact of Design Decisions in Budget-Constrained Agentic LLM Search</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted in 2026 Language Resources and Evaluation Conference (LREC)
    </div>
    <details class="paper-abstract">
      Agentic Retrieval-Augmented Generation (RAG) systems combine iterative search, planning prompts, and retrieval backends, but deployed settings impose explicit budgets on tool calls and completion tokens. We present a controlled measurement study of how search depth, retrieval strategy, and completion budget affect accuracy and cost under fixed constraints. Using Budget-Constrained Agentic Search (BCAS), a model-agnostic evaluation harness that surfaces remaining budget and gates tool use, we run comparisons across six LLMs and three question-answering benchmarks. Across models and datasets, accuracy improves with additional searches up to a small cap, hybrid lexical and dense retrieval with lightweight re-ranking produces the largest average gains in our ablation grid, and larger completion budgets are most helpful on HotpotQA-style synthesis. These results provide practical guidance for configuring budgeted agentic retrieval pipelines and are accompanied by reproducible prompts and evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08852v1">LDP: An Identity-Aware Protocol for Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 16 pages, 9 figures, 8 tables, 4 appendices
    </div>
    <details class="paper-abstract">
      As multi-agent AI systems grow in complexity, the protocols connecting them constrain their capabilities. Current protocols such as A2A and MCP do not expose model-level properties as first-class primitives, ignoring properties fundamental to effective delegation: model identity, reasoning profile, quality calibration, and cost characteristics. We present the LLM Delegate Protocol (LDP), an AI-native communication protocol introducing five mechanisms: (1) rich delegate identity cards with quality hints and reasoning profiles; (2) progressive payload modes with negotiation and fallback; (3) governed sessions with persistent context; (4) structured provenance tracking confidence and verification status; (5) trust domains enforcing security boundaries at the protocol level. We implement LDP as a plugin for the JamJet agent runtime and evaluate against A2A and random baselines using local Ollama models and LLM-as-judge evaluation. Identity-aware routing achieves ~12x lower latency on easy tasks through delegate specialization, though it does not improve aggregate quality in our small delegate pool; semantic frame payloads reduce token count by 37% (p=0.031) with no observed quality loss; governed sessions eliminate 39% token overhead at 10 rounds; and noisy provenance degrades synthesis quality below the no-provenance baseline, arguing that confidence metadata is harmful without verification. Simulated analyses show architectural advantages in attack detection (96% vs. 6%) and failure recovery (100% vs. 35% completion). This paper contributes a protocol design, reference implementation, and initial evidence that AI-native protocol primitives enable more efficient and governable delegation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08849v1">Investigating the Effects of LLM Use on Critical Thinking Under Time Constraints: Access Timing and Time Availability</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 CHI 2026
    </div>
    <details class="paper-abstract">
      The impact of large language models (LLMs) on critical thinking has provoked growing attention, yet this impact on actual performance may not be uniformly negative or positive. Particularly, the role of time -- the temporal context under which an LLM is provided -- remains overlooked. In a between-subjects experiment (n=393), we examined two types of time constraints for a critical thinking task requiring participants to make a reasoned decision for a real-world scenario based on diverse documents: (1) LLM access timing -- an LLM available only at the beginning (early), throughout (continuous), near the end (late), or not at all (no LLM), and (2) time availability -- insufficient or sufficient time for the task. We found a temporal reversal: LLM access from the start (early, continuous) improved performance under time pressure but impaired it with sufficient time, whereas beginning the task independently (late, no LLM) showed the opposite pattern. These findings demonstrate that time constraints fundamentally shape whether an LLM augments or undermines critical thinking, making time a central consideration when designing LLM support and evaluating human-AI collaboration in cognitive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08660v1">How Far Can Unsupervised RLVR Scale LLM Training?</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted to the ICLR 2026
    </div>
    <details class="paper-abstract">
      Unsupervised reinforcement learning with verifiable rewards (URLVR) offers a pathway to scale LLM training beyond the supervision bottleneck by deriving rewards without ground truth labels. Recent works leverage model intrinsic signals, showing promising early gains, yet their potential and limitations remain unclear. In this work, we revisit URLVR and provide a comprehensive analysis spanning taxonomy, theory and extensive experiments. We first classify URLVR methods into intrinsic versus external based on reward sources, then establish a unified theoretical framework revealing that all intrinsic methods converge toward sharpening the model's initial distribution This sharpening mechanism succeeds when initial confidence aligns with correctness but fails catastrophically when misaligned. Through systematic experiments, we show intrinsic rewards consistently follow a rise-then-fall pattern across methods, with collapse timing determined by model prior rather than engineering choices. Despite these scaling limits, we find intrinsic rewards remain valuable in test-time training on small datasets, and propose Model Collapse Step to measure model prior, serving as a practical indicator for RL trainability. Finally, we explore external reward methods that ground verification in computational asymmetries, showing preliminary evidence they may escape the confidence-correctness ceiling. Our findings chart boundaries for intrinsic URLVR while motivating paths toward scalable alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08604v1">What to Make Sense of in the Era of LLM? A Perspective from the Structure and Efforts in Sensemaking</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 CHI 2024 Sensemaking Workshop https://sites.google.com/view/chi2024-sensemaking-workshop/home?pli=1
    </div>
    <details class="paper-abstract">
      Sensemaking tasks often entail navigating through complex, ambiguous data to construct coherent insights. Prior work has shown that crowds can effectively distribute cognitive load, pooling diverse perspectives to enhance analytical depth. Recent advancements in LLMs have further expanded the toolkit for sensemaking, offering scalable data processing, complex pattern recognition, and the ability to infer and propose meaningful hypotheses. In this study, we explore how LLMs (i.e., GPT-4) can assist in a complex sensemaking task of deciphering fictional terrorist plots. We explore two different approaches for leveraging GPT-4's capabilities: a holistic sensemaking process and a step-by-step approach. Our preliminary investigations open the doors for future research into optimizing human-AI collaborative workflows, aiming to harness the complementary strengths of both for more effective sensemaking in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08520v1">SCAFFOLD-CEGIS: Preventing Latent Security Degradation in LLM-Driven Iterative Code Refinement</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      The application of large language models to code generation has evolved from one-shot generation to iterative refinement, yet the evolution of security throughout iteration remains insufficiently understood. Through comparative experiments on three mainstream LLMs, this paper reveals the iterative refinement paradox: specification drift during multi-objective optimization causes security to degrade gradually over successive iterations. Taking GPT-4o as an example, 43.7 % of iteration chains contain more vulnerabilities than the baseline after ten rounds, and cross-model experiments show that this phenomenon is prevalent. Further analysis shows that simply introducing static application security testing (SAST) gating cannot effectively suppress degradation; instead, it increases the latent security degradation rate from 12.5% under the unprotected baseline to 20.8 %. The root cause is that static-analysis rules cannot cover structural degradations such as the removal of defensive logic or the weakening of exception handling. To address this problem, we propose the SCAFFOLD-CEGIS framework. Drawing on the counterexample-guided inductive synthesis (CEGIS) paradigm, the framework adopts a multi-agent collaborative architecture that transforms security constraints from implicit prompts into explicit verifiable constraints. It automatically identifies and solidifies security-critical elements as hard constraints through semantic anchoring, enforces safety monotonicity through four-layer gated verification, and continuously assimilates experience from failures. Comparative experiments against six existing defense methods show that the full framework reduces the latent security degradation rate to 2.1% and achieves a safety monotonicity rate of 100%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.06046v4">Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 27 pages, 9 pages main text
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in the domains of medicine and public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, while there are a number of LLM benchmarks in the medical domain, currently little is known about LLM knowledge within the field of public health. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest proprietary LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, while there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08475v1">R2F: Repurposing Ray Frontiers for LLM-free Object Navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Zero-shot open-vocabulary object navigation has progressed rapidly with the emergence of large Vision-Language Models (VLMs) and Large Language Models (LLMs), now widely used as high-level decision-makers instead of end-to-end policies. Although effective, such systems often rely on iterative large-model queries at inference time, introducing latency and computational overhead that limit real-time deployment. To address this problem, we repurpose ray frontiers (R2F), a recently proposed frontier-based exploration paradigm, to develop an LLM-free framework for indoor open-vocabulary object navigation. While ray frontiers were originally used to bias exploration using semantic cues carried along rays, we reinterpret frontier regions as explicit, direction-conditioned semantic hypotheses that serve as navigation goals. Language-aligned features accumulated along out-of-range rays are stored sparsely at frontiers, where each region maintains multiple directional embeddings encoding plausible unseen content. In this way, navigation then reduces to embedding-based frontier scoring and goal tracking within a classical mapping and planning pipeline, eliminating iterative large-model reasoning. We further introduce R2F-VLN, a lightweight extension for free-form language instructions using syntactic parsing and relational verification without additional VLM or LLM components. Experiments in Habitat-sim and on a real robotic platform demonstrate competitive state-of-the-art zero-shot performance with real-time execution, achieving up to 6 times faster runtime than VLM-based alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08443v1">LLM-Driven Online Aggregation for Unstructured Text Analytics</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 DASFAA 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong capabilities in text processing, and recent research has augmented SQL and DataFrame with LLM-powered semantic operators for data analysis. However, LLM-based data processing is hindered by slower token generation speeds compared to relational queries. To enhance real-time responsiveness, we propose OLLA, an LLM-driven online aggregation framework that accelerates semantic processing within relational queries. In contrast to batch-processing systems that yield results only after the entire dataset is processed, our approach incrementally transforms text into a structured data stream and applies online aggregation to provide progressive output. To enhance our online aggregation process, we introduce a semantic stratified sampling approach that improves data selection and expedites convergence to the ground truth. Evaluations show that OLLA reaches 1% accuracy error bound compared with labeled ground truth using less than 4% of the full-data time. It achieves speedups ranging from 1.6$\times$ to 38$\times$ across diverse domains, measured by comparing the time to reach a 5% error bound with that of full-data time. We release our code at https://github.com/olla-project/llm-online-agg.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08429v1">One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      LLM agents that retrieve external knowledge typically generate a search query as text, then run a separate embedding model to encode it into a vector. This two-model pipeline adds infrastructure complexity and latency, yet is redundant: the LLM already encodes the full conversational context in its hidden states. We propose equipping LLM agents with native retrieval capability by adding a lightweight projection head that maps hidden states directly into the embedding space, eliminating the need for a separate embedding model. Trained with a combination of alignment, contrastive, and rank distillation losses, our method retains 97\% of baseline retrieval quality while enabling the LLM agent to search with its own representations. Experiments on the QReCC conversational search benchmark show competitive Recall@10 and MRR@10 compared to the standard generate-then-encode pipeline, with systematic ablations confirming the contribution of each loss component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08388v1">A Hierarchical Error-Corrective Graph Framework for Autonomous Agents with LLM-Based Action Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      We propose a Hierarchical Error-Corrective Graph FrameworkforAutonomousAgentswithLLM-BasedActionGeneration(HECG),whichincorporates three core innovations: (1) Multi-Dimensional Transferable Strategy (MDTS): by integrating task quality metrics (Q), confidence/cost metrics (C), reward metrics (R), and LLM-based semantic reasoning scores (LLM-Score), MDTS achieves multi-dimensional alignment between quantitative performance and semantic context, enabling more precise selection of high-quality candidate strate gies and effectively reducing the risk of negative transfer. (2) Error Matrix Classification (EMC): unlike simple confusion matrices or overall performance metrics, EMC provides structured attribution of task failures by categorizing errors into ten types, such as Strategy Errors (Strategy Whe) and Script Parsing Errors (Script-Parsing-Error), and decomposing them according to severity, typical actions, error descriptions, and recoverability. This allows precise analysis of the root causes of task failures, offering clear guidance for subsequent error correction and strategy optimization rather than relying solely on overall success rates or single performance metrics. (3) Causal-Context Graph Retrieval (CCGR): to enhance agent retrieval capabilities in dynamic task environments, we construct graphs from historical states, actions, and event sequences, where nodes store executed actions, next-step actions, execution states, transferable strategies, and other relevant information, and edges represent causal dependencies such as preconditions for transitions between nodes. CCGR identifies subgraphs most relevant to the current task context, effectively capturing structural relationships beyond vector similarity, allowing agents to fully leverage contextual information, accelerate strategy adaptation, and improve execution reliability in complex, multi-step tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13606v2">LaVCa: LLM-assisted Visual Cortex Captioning</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted to ICLR 2026. Website: https://sites.google.com/view/lavca-llm/
    </div>
    <details class="paper-abstract">
      Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15656v2">PhishDebate: An LLM-Based Multi-Agent Framework for Phishing Website Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Please cite as: W. Li, S. Manickam, Y. -W. Chong and S. Karuppayah, "PhishDebate: An LLM-Based Multi-Agent Framework for Phishing Website Detection," 2025 IEEE International Conference on Big Data (BigData), Macau, China, 2025, pp. 6606-6615, doi: 10.1109/BigData66926.2025.11401440
    </div>
    <details class="paper-abstract">
      Phishing websites remain a major cybersecurity threat, exploiting deceptive structures, brand impersonation, and social engineering to evade detection. Recent advances in large language models (LLMs) have improved phishing detection through contextual understanding, yet most existing approaches rely on single-agent classification, which is prone to hallucination and often lacks interpretability and robustness. To address these limitations, we propose PhishDebate, a modular multi-agent LLM-based debate framework for phishing website detection. Four specialized agents independently analyze webpage aspects, including URL structure, HTML composition, semantic content, and brand impersonation, under the coordination of a Moderator and final Judge. Through structured debate and divergent reasoning, the framework achieves more accurate and interpretable decisions. By reducing uncertain predictions and providing transparent reasoning, PhishDebate functions as an analyst-augmentation system that lowers cognitive load and supports early, left-of-exploit detection of phishing threats. Evaluations on commercial LLMs show that PhishDebate achieves 98.2 % recall on a real-world phishing dataset and outperforms single-agent and Chain-of-Thought (CoT) baselines. Its modular design enables agent-level configurability, allowing adaptation to varying resource and application requirements, and offers scalability to high-velocity, large-scale security data environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08286v1">LAMUS: A Large-Scale Corpus for Legal Argument Mining from U.S. Caselaw using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Legal argument mining aims to identify and classify the functional components of judicial reasoning, such as facts, issues, rules, analysis, and conclusions. Progress in this area is limited by the lack of large-scale, high-quality annotated datasets for U.S. caselaw, particularly at the state level. This paper introduces LAMUS, a sentence-level legal argument mining corpus constructed from U.S. Supreme Court decisions and Texas criminal appellate opinions. The dataset is created using a data-centric pipeline that combines large-scale case collection, LLM-based automatic annotation, and targeted human-in-the-loop quality refinement. We formulate legal argument mining as a six-class sentence classification task and evaluate multiple general-purpose and legal-domain language models under zero-shot, few-shot, and chain-of-thought prompting strategies, with LegalBERT as a supervised baseline. Results show that chain-of-thought prompting substantially improves LLM performance, while domain-specific models exhibit more stable zero-shot behavior. LLM-assisted verification corrects nearly 20% of annotation errors, improving label consistency. Human verification achieves Cohen's Kappa of 0.85, confirming annotation quality. LAMUS provides a scalable resource and empirical insights for future legal NLP research. All code and datasets can be accessed for reproducibility on GitHub at: https://github.com/LavanyaPobbathi/LAMUS/tree/main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08274v1">How Much Do LLMs Hallucinate in Document Q&A Scenarios? A 172-Billion-Token Study Across Temperatures, Context Lengths, and Hardware Platforms</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 18 pages, 12 tables, 2 figures
    </div>
    <details class="paper-abstract">
      How much do large language models actually hallucinate when answering questions grounded in provided documents? Despite the critical importance of this question for enterprise AI deployments, reliable measurement has been hampered by benchmarks that rely on static datasets vulnerable to contamination, LLM-based judges with documented biases, or evaluation scales too small for statistical confidence. We address this gap using RIKER, a ground-truth-first evaluation methodology that enables deterministic scoring without human annotation. Across 35 open-weight models, three context lengths (32K, 128K, and 200K tokens), four temperature settings, and three hardware platforms (NVIDIA H200, AMD MI300X, and Intel Gaudi 3), we conducted over 172 billion tokens of evaluation - an order of magnitude beyond prior work. Our findings reveal that: (1) even the best-performing models fabricate answers at a non-trivial rate - 1.19% at best at 32K, with top-tier models at 5 - 7% - and fabrication rises steeply with context length, nearly tripling at 128K and exceeding 10% for all models at 200K; (2) model selection dominates all other factors, with overall accuracy spanning a 72-percentage-point range and model family predicting fabrication resistance better than model size; (3) temperature effects are nuanced - T=0.0 yields the best overall accuracy in roughly 60% of cases, but higher temperatures reduce fabrication for the majority of models and dramatically reduce coherence loss (infinite generation loops), which can reach 48x higher rates at T=0.0 versus T=1.0; (4) grounding ability and fabrication resistance are distinct capabilities - models that excel at finding facts may still fabricate facts that do not exist; and (5) results are consistent across hardware platforms, confirming that deployment decisions need not be hardware-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08262v1">FinToolBench: Evaluating LLM Agents for Real-World Financial Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into the financial domain is driving a paradigm shift from passive information retrieval to dynamic, agentic interaction. While general-purpose tool learning has witnessed a surge in benchmarks, the financial sector, characterized by high stakes, strict compliance, and rapid data volatility, remains critically underserved. Existing financial evaluations predominantly focus on static textual analysis or document-based QA, ignoring the complex reality of tool execution. Conversely, general tool benchmarks lack the domain-specific rigor required for finance, often relying on toy environments or a negligible number of financial APIs. To bridge this gap, we introduce FinToolBench, the first real-world, runnable benchmark dedicated to evaluating financial tool learning agents. Unlike prior works limited to a handful of mock tools, FinToolBench establishes a realistic ecosystem coupling 760 executable financial tools with 295 rigorous, tool-required queries. We propose a novel evaluation framework that goes beyond binary execution success, assessing agents on finance-critical dimensions: timeliness, intent type, and regulatory domain alignment. Furthermore, we present FATR, a finance-aware tool retrieval and reasoning baseline that enhances stability and compliance. By providing the first testbed for auditable, agentic financial execution, FinToolBench sets a new standard for trustworthy AI in finance. The tool manifest, execution environment, and evaluation code will be open-sourced to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08256v1">NCL-UoR at SemEval-2026 Task 5: Embedding-Based Methods, Fine-Tuning, and LLMs for Word Sense Plausibility Rating</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Word sense plausibility rating requires predicting the human-perceived plausibility of a given word sense on a 1--5 scale in the context of short narrative stories containing ambiguous homonyms. This paper systematically compares three approaches: (1) embedding-based methods pairing sentence embeddings with standard regressors, (2) transformer fine-tuning with parameter-efficient adaptation, and (3) large language model (LLM) prompting with structured reasoning and explicit decision rules. The best-performing system employs a structured prompting strategy that decomposes evaluation into narrative components (precontext, target sentence, ending) and applies explicit decision rules for rating calibration. The analysis reveals that structured prompting with decision rules substantially outperforms both fine-tuned models and embedding-based approaches, and that prompt design matters more than model scale for this task. The code is publicly available at https://github.com/tongwu17/SemEval-2026-Task5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08234v1">The Struggle Between Continuation and Refusal: A Mechanistic Analysis of the Continuation-Triggered Jailbreak in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), the safety of LLMs has become a critical concern. Despite significant efforts in safety alignment, current LLMs remain vulnerable to jailbreaking attacks. However, the root causes of such vulnerabilities are still poorly understood, necessitating a rigorous investigation into jailbreak mechanisms across both academic and industrial communities. In this work, we focus on a continuation-triggered jailbreak phenomenon, whereby simply relocating a continuation-triggered instruction suffix can substantially increase jailbreak success rates. To uncover the intrinsic mechanisms of this phenomenon, we conduct a comprehensive mechanistic interpretability analysis at the level of attention heads. Through causal interventions and activation scaling, we show that this jailbreak behavior primarily arises from an inherent competition between the model's intrinsic continuation drive and the safety defenses acquired through alignment training. Furthermore, we perform a detailed behavioral analysis of the identified safety-critical attention heads, revealing notable differences in the functions and behaviors of safety heads across different model architectures. These findings provide a novel mechanistic perspective for understanding and interpreting jailbreak behaviors in LLMs, offering both theoretical insights and practical implications for improving model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22159v2">RedSage: A Cybersecurity Generalist LLM</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Published at ICLR 2026; Project page: https://risys-lab.github.io/RedSage/
    </div>
    <details class="paper-abstract">
      Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation pipeline that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised fine-tuning. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a benchmark with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the baseline models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00086v3">Impact of LLMs news Sentiment Analysis on Stock Price Movement Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      This paper addresses stock price movement prediction by leveraging LLM-based news sentiment analysis. Earlier works have largely focused on proposing and assessing sentiment analysis models and stock movement prediction methods, however, separately. Although promising results have been achieved, a clear and in-depth understanding of the benefit of the news sentiment to this task, as well as a comprehensive assessment of different architecture types in this context, is still lacking. Herein, we conduct an evaluation study that compares 3 different LLMs, namely, DeBERTa, RoBERTa and FinBERT, for sentiment-driven stock prediction. Our results suggest that DeBERTa outperforms the other two models with an accuracy of 75% and that an ensemble model that combines the three models can increase the accuracy to about 80%. Also, we see that sentiment news features can benefit (slightly) some stock market prediction models, i.e., LSTM-, PatchTST- and tPatchGNN-based classifiers and PatchTST- and TimesNet-based regression tasks models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01853v2">Security and Quality in LLM-Generated Code: A Multi-Language, Multi-Model Analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 20 pages, 13 tables. Accepted to IEEE Transactions on Dependable and Secure Computing
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI)-driven code generation tools are increasingly used throughout the software development lifecycle to accelerate coding tasks. However, the security of AI-generated code using Large Language Models (LLMs) remains underexplored, with studies revealing various risks and weaknesses. This paper analyzes the security of code generated by LLMs across different programming languages. We introduce a dataset of 200 tasks grouped into six categories to evaluate the performance of LLMs in generating secure and maintainable code. Our research shows that while LLMs can automate code creation, their security effectiveness varies by language. Many models fail to utilize modern security features in recent compiler and toolkit updates, such as Java 17. Moreover, outdated methods are still commonly used, particularly in C++. This highlights the need for advancing LLMs to enhance security and quality while incorporating emerging best practices in programming languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17262v4">Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted by The Fourteenth International Conference on Learning Representations (ICLR2026)
    </div>
    <details class="paper-abstract">
      The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for comprehensive understanding of scaling properties. This is challenged by: 1) the emergence phenomenon, where unpredictable capabilities appearing suddenly at critical model scales; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby constructing a more stable and predictable task subset that exhibits well-behaved scaling characteristics with the increase of compute budget. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.55\% average prediction error across eight key LLM benchmarks, thus providing actionable insights for scaling properties and training monitoring during LLM pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08185v1">SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a prevailing technique for deploying large language models (LLMs) efficiently in terms of both memory and computation, across edge devices and server platforms. Existing PTQ methods primarily aim to reduce precision in weights and activations by mitigating quantization errors caused by channel-wise outlier activations (e.g., pre-quantization scaling, online transformations, or low-rank error reconstruction). Among these approaches, error reconstruction with low-rank adaptation (LoRA) has proven particularly effective, as it introduces a lightweight auxiliary computation path without requiring heavy optimization or additional online layers. However, prior studies reveal severe accuracy degradation under W4A4 settings, and conventional low-rank adaptations rely on two sequential factors, necessitating intermediate quantization during inference and thereby limiting low-precision efficiency. In this work, we propose SERQ, a saliency-aware error reconstruction method for low-bit LLM inference that employs a single low-rank compensation matrix. SERQ preserves efficient 4-bit matrix multiplication in linear layers by jointly mitigating quantization errors arising from both activation and weight saliency through three stages: (1) static activation flattening, (2) saliency-aware error reconstruction, and (3) offline weight permutation. The method incurs additional computation only for low-rank error reconstruction via a single decomposition, while all other operations are performed offline, thereby keeping latency overhead minimal. Empirically, SERQ outperforms prior error reconstruction methods under both W4A8 and W4A4 settings, and achieves higher accuracy than state-of-the-art rotation-based W4A4 approaches, while substantially reducing calibration complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08182v1">TildeOpen LLM: Leveraging Curriculum Learning to Achieve Equitable Language Representation</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 LREC 2026
    </div>
    <details class="paper-abstract">
      Large language models often underperform in many European languages due to the dominance of English and a few high-resource languages in training data. This paper presents TildeOpen LLM, a 30-billion-parameter open-weight foundational model trained for 34 European languages to promote linguistic equity and improve performance for low-resource languages. To address the data imbalance, we combine dataset upsampling with a curriculum-based training schedule that alternates between uniform and natural language distributions. The resulting model performs favorably compared to other multilingual LLMs despite being trained with significantly fewer computing resources. Evaluation across multiple multilingual benchmarks shows that TildeOpen surpasses existing open-weight models in text generation and comprehension, particularly for Baltic, Finno-Ugric, and Slavic languages. Human evaluations confirm an up to tenfold reduction in linguistic errors relative to leading baselines. The model and associated resources are fully open-weight and publicly available at huggingface.co/TildeAI/TildeOpen-30b. These outcomes demonstrate that careful data curation and balanced training strategies can substantially enhance multilingual model quality without increasing model size or training volume.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08181v1">AutoAdapt: An Automated Domain Adaptation Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in open domains but struggle in specialized settings with limited data and evolving knowledge. Existing domain adaptation practices rely heavily on manual trial-and-error processes, incur significant hyperparameter complexity, and are highly sensitive to data and user preferences, all under the high cost of LLM training. Moreover, the interactions and transferability of hyperparameter choices across models/domains remain poorly understood, making adaptation gains uncertain even with substantial effort. To solve these challenges, we present AutoAdapt, a novel end-to-end automated framework for efficient and reliable LLM domain adaptation. AutoAdapt leverages curated knowledge bases from literature and open-source resources to reduce expert intervention. To narrow the search space, we design a novel multi-agent debating system in which proposal and critic agents iteratively interact to align user intent and incorporate data signals and best practices into the planning process. To optimize hyperparameters under tight budgets, we propose AutoRefine, a novel LLM-based surrogate that replaces costly black-box search. Across 10 tasks, AutoAdapt achieves a 25% average relative accuracy improvement over state-of-the-art Automated Machine Learning baselines with minimal overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05500v1">POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Technical report v1 (14 pages, 7 figures, project page: https://spherelab.ai/poetx/)
    </div>
    <details class="paper-abstract">
      Efficient and stable training of large language models (LLMs) remains a core challenge in modern machine learning systems. To address this challenge, Reparameterized Orthogonal Equivalence Training (POET), a spectrum-preserving framework that optimizes each weight matrix through orthogonal equivalence transformation, has been proposed. Although POET provides strong training stability, its original implementation incurs high memory consumption and computational overhead due to intensive matrix multiplications. To overcome these limitations, we introduce POET-X, a scalable and memory-efficient variant that performs orthogonal equivalence transformations with significantly reduced computational cost. POET-X maintains the generalization and stability benefits of POET while achieving substantial improvements in throughput and memory efficiency. In our experiments, POET-X enables the pretraining of billion-parameter LLMs on a single Nvidia H100 GPU, and in contrast, standard optimizers such as AdamW run out of memory under the same settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05494v1">Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large language models sometimes produce false or misleading responses. Two approaches to this problem are honesty elicitation -- modifying prompts or weights so that the model answers truthfully -- and lie detection -- classifying whether a given response is false. Prior work evaluates such methods on models specifically trained to lie or conceal information, but these artificial constructions may not resemble naturally-occurring dishonesty. We instead study open-weights LLMs from Chinese developers, which are trained to censor politically sensitive topics: Qwen3 models frequently produce falsehoods about subjects like Falun Gong or the Tiananmen protests while occasionally answering correctly, indicating they possess knowledge they are trained to suppress. Using this as a testbed, we evaluate a suite of elicitation and lie detection techniques. For honesty elicitation, sampling without a chat template, few-shot prompting, and fine-tuning on generic honesty data most reliably increase truthful responses. For lie detection, prompting the censored model to classify its own responses performs near an uncensored-model upper bound, and linear probes trained on unrelated data offer a cheaper alternative. The strongest honesty elicitation techniques also transfer to frontier open-weights models including DeepSeek R1. Notably, no technique fully eliminates false responses. We release all prompts, code, and transcripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05489v1">NL2GDS: LLM-aided interface for Open Source Chip Design</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The growing complexity of hardware design and the widening gap between high-level specifications and register-transfer level (RTL) implementation hinder rapid prototyping and system design. We introduce NL2GDS (Natural Language to Layout), a novel framework that leverages large language models (LLMs) to translate natural language hardware descriptions into synthesizable RTL and complete GDSII layouts via the open-source OpenLane ASIC flow. NL2GDS employs a modular pipeline that captures informal design intent, generates HDL using multiple LLM engines and verifies them, and orchestrates automated synthesis and layout. Evaluations on ISCAS'85 and ISCAS'89 benchmark designs demonstrate up to 36% area reduction, 35% delay reduction, and 70% power savings compared to baseline designs, highlighting its potential to democratize ASIC design and accelerate hardware innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05485v1">Towards Provably Unbiased LLM Judges via Bias-Bounded Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      As AI models progress beyond simple chatbots into more complex workflows, we draw ever closer to the event horizon beyond which AI systems will be utilized in autonomous, self-maintaining feedback loops. Any autonomous AI system will depend on automated, verifiable rewards and feedback; in settings where ground truth is sparse or non-deterministic, one practical source of such rewards is an LLM-as-a-Judge. Although LLM judges continue to improve, the literature has yet to introduce systems capable of enforcing standards with strong guarantees, particularly when bias vectors are unknown or adversarially discovered. To remedy this issue, we propose average bias-boundedness (A-BB), an algorithmic framework which formally guarantees reductions of harm/impact as a result of any measurable bias in an LLM judge. Evaluating on Arena-Hard-Auto with four LLM judges, we achieve (tau=0.5, delta=0.01) bias-bounded guarantees while retaining 61-99% correlation with original rankings across formatting and schematic bias settings, with most judge-bias combinations exceeding 80%. The code to reproduce our findings is available at https://github.com/penfever/bias-bounded-evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05471v1">Leveraging LLM Parametric Knowledge for Fact Checking without Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Trustworthiness is a core research challenge for agentic AI systems built on Large Language Models (LLMs). To enhance trust, natural language claims from diverse sources, including human-written text, web content, and model outputs, are commonly checked for factuality by retrieving external knowledge and using an LLM to verify the faithfulness of claims to the retrieved evidence. As a result, such methods are constrained by retrieval errors and external data availability, while leaving the models intrinsic fact-verification capabilities largely unused. We propose the task of fact-checking without retrieval, focusing on the verification of arbitrary natural language claims, independent of their source. To study this setting, we introduce a comprehensive evaluation framework focused on generalization, testing robustness to (i) long-tail knowledge, (ii) variation in claim sources, (iii) multilinguality, and (iv) long-form generation. Across 9 datasets, 18 methods and 3 models, our experiments indicate that logit-based approaches often underperform compared to those that leverage internal model representations. Building on this finding, we introduce INTRA, a method that exploits interactions between internal representations and achieves state-of-the-art performance with strong generalization. More broadly, our work establishes fact-checking without retrieval as a promising research direction that can complement retrieval-based frameworks, improve scalability, and enable the use of such systems as reward signals during training or as components integrated into the generation process.
    </details>
</div>
