# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04288v2">LLM-Enabled Open-Source Systems in the Wild: An Empirical Study of Vulnerabilities in GitHub Security Advisories</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 The 2nd International Workshop on Large Language Model Supply Chain Analysis (LLMSC 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded in open-source software (OSS) ecosystems, creating complex interactions among natural language prompts, probabilistic model outputs, and execution-capable components. However, it remains unclear whether traditional vulnerability disclosure frameworks adequately capture these model-mediated risks. To investigate this, we analyze 295 GitHub Security Advisories published between January 2025 and January 2026 that reference LLM-related components, and we manually annotate a sample of 100 advisories using the OWASP Top 10 for LLM Applications 2025. We find no evidence of new implementation-level weakness classes specific to LLM systems. Most advisories map to established CWEs, particularly injection and deserialization weaknesses. At the same time, the OWASP-based analysis reveals recurring architectural risk patterns, especially Supply Chain, Excessive Agency, and Prompt Injection, which often co-occur across multiple stages of execution. These results suggest that existing advisory metadata captures code-level defects but underrepresents model-mediated exposure. We conclude that combining the CWE and OWASP perspectives provides a more complete and necessary view of vulnerabilities in LLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14237v1">TOPCELL: Topology Optimization of Standard Cell via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Accepted to the 63rd ACM/IEEE Design Automation Conference (DAC 2026). 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Transistor topology optimization is a critical step in standard cell design, directly dictating diffusion sharing efficiency and downstream routability. However, identifying optimal topologies remains a persistent bottleneck, as conventional exhaustive search methods become computationally intractable with increasing circuit complexity in advanced nodes. This paper introduces TOPCELL, a novel and scalable framework that reformulates high-dimensional topology exploration as a generative task using Large Language Models (LLMs). We employ Group Relative Policy Optimization (GRPO) to fine-tune the model, aligning its topology optimization strategy with logical (circuit) and spatial (layout) constraints. Experimental results within an industrial flow targeting an advanced 2nm technology node demonstrate that TOPCELL significantly outperforms foundation models in discovering routable, physically-aware topologies. When integrated into a state-of-the-art (SOTA) automation flow for a 7nm library generation task, TOPCELL exhibits robust zero-shot generalization and matches the layout quality of exhaustive solvers while achieving an 85.91x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13353v1">Cross-Domain Query Translation for Network Troubleshooting: A Multi-Agent LLM Framework with Privacy Preservation and Self-Reflection</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted for publication in EuCNC & 6G Summit 2026. \c{opyright} 2026 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission
    </div>
    <details class="paper-abstract">
      This paper presents a hierarchical multi-agent LLM architecture to bridge communication gaps between non-technical end users and telecommunications domain experts in private network environments. We propose a cross-domain query translation framework that leverages specialized language models coordinated through multi-agent reflection-based reasoning. The resulting system addresses three critical challenges: (1) accurately classify user queries related to telecommunications network issues using a dual-stage hierarchical approach, (2) preserve user privacy through the anonymization of semantically relevant personally identifiable information (PII) while maintaining diagnostic utility, and (3) translate technical expert responses into user-comprehensible language. Our approach employs ReAct-style agents enhanced with self-reflection mechanisms for iterative output refinement, semantic-preserving anonymization techniques respecting $k$-anonymity and differential privacy principles, and few-shot learning strategies designed for limited training data scenarios. The framework was comprehensively evaluated on 10,000 previously unseen validation scenarios across various vertical industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13349v1">When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messages through full key-value (KV) caches. However, full KV relay incurs high memory and communication cost. We adapt eviction-style KV compression to this setting and introduce Orthogonal Backfill (OBF) to mitigate information loss from hard eviction. OBF injects a low-rank orthogonal residual from discarded KV states into the retained KV states. We evaluate proposed method against full KV relay on nine standard benchmarks spanning mathematical reasoning, coding, and knowledge-intensive QA. It achieves performance comparable to full KV relay while reducing communication cost by 79.8%--89.4%. OBF further improves the performance and achieves the best results on 7 of the 9 benchmarks. This suggests that more information does not necessarily lead to better communication; preserving the most useful information matters more. Our codebase is publicly available on https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.03610v3">Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are reshaping the game industry, by enabling more intelligent and human-preferable characters. Yet, current game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets to adapt pre-trained LLMs into gaming agents. To fill these gaps, we present Orak, a benchmark for training and evaluating LLM agents across 12 popular video games spanning all major genres. Using a plug-and-play interface built on Model Context Protocol (MCP), Orak supports systematic and reproducible studies of agentic modules in varied game scenarios. We further release a fine-tuning dataset of expert LLM gameplay trajectories covering multiple genres, turning general LLMs into effective game agents. Orak offers a united evaluation framework, including game leaderboards, LLM battle arenas, and \fix{ablation studies} of input modality, agentic strategies, and fine-tuning effects, establishing a foundation towards versatile gaming agents. Code and datasets are available at https://github.com/krafton-ai/Orak and https://huggingface.co/datasets/KRAFTON/Orak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13328v1">Multi-Task LLM with LoRA Fine-Tuning for Automated Cancer Staging and Biomarker Extraction</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 11 pages, 3 figures and 4 tables in the main manuscript. Additional content, figures and tables are in supplementary material section. 17 pages in total
    </div>
    <details class="paper-abstract">
      Pathology reports serve as the definitive record for breast cancer staging, yet their unstructured format impedes large-scale data curation. While Large Language Models (LLMs) offer semantic reasoning, their deployment is often limited by high computational costs and hallucination risks. This study introduces a parameter-efficient, multi-task framework for automating the extraction of Tumor-Node-Metastasis (TNM) staging, histologic grade, and biomarkers. We fine-tune a Llama-3-8B-Instruct encoder using Low-Rank Adaptation (LoRA) on a curated, expert-verified dataset of 10,677 reports. Unlike generative approaches, our architecture utilizes parallel classification heads to enforce consistent schema adherence. Experimental results demonstrate that the model achieves a Macro F1 score of 0.976, successfully resolving complex contextual ambiguities and heterogeneous reporting formats that challenge traditional extraction methods including rule-based natural language processing (NLP) pipelines, zero-shot LLMs, and single-task LLM baselines. The proposed adapter-efficient, multi-task architecture enables reliable, scalable pathology-derived cancer staging and biomarker profiling, with the potential to enhance clinical decision support and accelerate data-driven oncology research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07053v2">Animating Petascale Time-varying Data on Commodity Hardware with LLM-assisted Scripting</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 ©2026 IEEE. Personal use of this material is permitted. 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. N.B. Due to the limitation "The abstract field cannot be longer than 1,920 characters", the abstract here is shorter than that in the original PDF file
    </div>
    <details class="paper-abstract">
      Scientists face significant visualization challenges as time-varying datasets grow in speed and volume, often requiring specialized infrastructure and expertise to handle massive datasets. Petascale climate models generated in NASA laboratories require a dedicated group of graphics and media experts and access to high-performance computing resources. Scientists may need to share scientific results with the community iteratively and quickly. However, the time-consuming trial-and-error process incurs significant data transfer overhead and far exceeds the time and resources allocated for typical post-analysis visualization tasks, disrupting the production workflow. Our paper introduces a user-friendly framework for creating 3D animations of petascale, time-varying data on a commodity workstation. Our contributions: (i) Generalized Animation Descriptor (GAD) with a keyframe-based adaptable abstraction for animation, (ii) efficient data access from cloud-hosted repositories to reduce data management overhead, (iii) tailored rendering system, and (iv) an LLM-assisted conversational interface as a scripting module to allow domain scientists with no visualization expertise to create animations of their region of interest. We demonstrate the framework's effectiveness with two case studies: first, by generating animations in which sampling criteria are specified based on prior knowledge, and second, by generating AI-assisted animations in which sampling parameters are derived from natural-language user prompts. In all cases, we use large-scale NASA climate-oceanographic datasets that exceed 1PB in size yet achieve a fast turnaround time of 1 minute to 2 hours. Users can generate a rough draft of the animation within minutes, then seamlessly incorporate as much high-resolution data as needed for the final version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15550v3">Common to Whom? Regional Cultural Commonsense and LLM Bias in India</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Existing cultural commonsense benchmarks treat nations as monolithic, assuming uniform practices within national boundaries. But does cultural commonsense hold uniformly within a nation, or does it vary at the sub-national level? We introduce Indica, the first benchmark designed to test LLMs' ability to address this question, focusing on India - a nation of 28 states, 8 union territories, and 22 official languages. We collect human-annotated answers from five Indian regions (North, South, East, West, and Central) across 515 questions spanning 8 domains of everyday life, yielding 1,630 region-specific question-answer pairs. Strikingly, only 39.4% of questions elicit agreement across all five regions, demonstrating that cultural commonsense in India is predominantly regional, not national. We evaluate eight state-of-the-art LLMs and find two critical gaps: models achieve only 13.4%-20.9% accuracy on region-specific questions, and they exhibit geographic bias, over-selecting Central and North India as the "default" (selected 30-40% more often than expected) while under-representing East and West. Beyond India, our methodology provides a generalizable framework for evaluating cultural commonsense in any culturally heterogeneous nation, from question design grounded in anthropological taxonomy, to regional data collection, to bias measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13298v1">Can Agents Secure Hardware? Evaluating Agentic LLM-Driven Obfuscation for IP Protection</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 5 pages, 3 figures,
    </div>
    <details class="paper-abstract">
      The globalization of integrated circuit (IC) design and manufacturing has increased the exposure of hardware intellectual property (IP) to untrusted stages of the supply chain, raising concerns about reverse engineering, piracy, tampering, and overbuilding. Hardware netlist obfuscation is a promising countermeasure, but automating the generation of functionally correct and security-relevant obfuscated circuits remains challenging, particularly for benchmark-scale designs. This paper presents an agentic, large language model (LLM)-driven framework for automated hardware netlist obfuscation. The proposed framework combines retrieval-grounded planning, structured lock-plan generation, deterministic netlist compilation, functional verification, and SAT-based security evaluation. Rather than a single prompt-to-output generation step, the framework decomposes the task into specialized stages for circuit analysis, synthesis, verification, and attack evaluation. We evaluate the framework on ISCAS-85 benchmarks using functional equivalence checking and SAT-based attacks. Results show that the framework generates correct locked netlists while introducing measurable output corruption under incorrect keys, while SAT attacks remain effective. These findings highlight both the potential and current limitations of agentic LLM-driven obfuscation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13271v1">Enhancing Confidence Estimation in Telco LLMs via Twin-Pass CoT-Ensembling</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to complex telecommunications tasks, including 3GPP specification analysis and O-RAN network troubleshooting. However, a critical limitation remains: LLM-generated confidence scores are often biased and unreliable, frequently exhibiting systematic overconfidence. This lack of trustworthy self-assessment makes it difficult to verify model outputs and safely rely on them in practice. In this paper, we study confidence calibration in telecom-domain LLMs using the representative Gemma-3 model family (4B, 12B, and 27B parameters), evaluated on TeleQnA, ORANBench, and srsRANBench. We show that standard single-pass, verbalized confidence estimates fail to reflect true correctness, often assigning high confidence to incorrect predictions. To address this, we propose a novel Twin-Pass Chain of Thought (CoT)-Ensembling methodology for improving confidence estimation by leveraging multiple independent reasoning evaluations and aggregating their assessments into a calibrated confidence score. Our approach reduces Expected Calibration Error (ECE) by up to 88% across benchmarks, significantly improving the reliability of model self-assessment. These results highlight the limitations of current confidence estimation practices and demonstrate a practical path toward more trustworthy evaluation of LLM outputs in telecommunications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22174v2">BitFlipScope: Scalable Fault Localization and Recovery for Bit-Flip Corruptions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted at the IEEE International Symposium on Hardware Oriented Security and Trust (HOST) 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed in practical and safety-critical settings are increasingly susceptible to bit-flip faults caused by hardware degradation, cosmic radiation, or deliberate fault-injection attacks such as Rowhammer. These faults silently corrupt internal parameters and can lead to unpredictable or dangerous model behavior. Localizing these corruptions is essential: without identifying the affected region, it is impossible to diagnose the source of degradation, apply targeted corrective measures, or restore model functionality without resorting to costly fine-tuning or full retraining. This work introduces BitFlipScope, a scalable, software-based framework for identifying fault-affected regions within transformer architectures under two deployment scenarios. When a clean reference model is available, BitFlipScope performs differential analysis of outputs, hidden states, and internal activations for detecting anomalous behavior indicative of corruption to pinpoint or localize faults. When no reference model exists, it uses residual-path perturbation and loss-sensitivity profiling to infer the fault-impacted region directly from the corrupted model. In both settings, the framework not only enables effective fault diagnosis but also supports lightweight performance recovery without fine-tuning, offering a practical path to restoring corrupted models. Together, these capabilities make BitFlipScope an important step toward trustworthy, fault-resilient LLM deployment in hardware-prone and adversarial environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12564v7">Sell Me This Stock: Unsafe Recommendation Drift in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      When a multi-turn LLM recommendation agent consumes incorrect tool data, it recommends unsuitable products while standard quality metrics stay near-perfect, a pattern we call evaluation blindness. We replay 23-turn financial advisory conversations across eight language models and find three counterintuitive failure modes. First, stronger models are not safer: the best-performing model has the highest quality score yet the worst suitability violations (99.1% of turns). This points to an alignment-grounding tension: the same property that makes it an effective agent, faithfully grounding its reasoning in tool data, makes it the most reliable executor of bad data. Across all models, 80% of risk-score citations repeat the manipulated value verbatim, and not a single turn out of 1,840 questions the tool outputs. Second, the failures are not cumulative: 95% of violations stem from the current turn's data rather than contamination building up in memory, meaning a single bad turn is enough to compromise safety. Third, while the model internally detects the manipulation (sparse autoencoder probing distinguishes adversarial from random perturbations), this awareness does not translate into safer output. Both representation-level interventions (recovering less than 6% of the gap) and prompt-level self-verification fail, as the agent ultimately relies on the same manipulated data. While incorporating suitability constraints into ranking metrics makes the gap visible, our findings suggest that safe deployment requires independent monitoring against a data source the agent cannot influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13243v1">Lazy or Efficient? Towards Accessible Eye-Tracking Event Detection Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Gaze event detection is fundamental to vision science, human-computer interaction, and applied analytics. However, current workflows often require specialized programming knowledge and careful handling of heterogeneous raw data formats. Classical detectors such as I-VT and I-DT are effective but highly sensitive to preprocessing and parameterization, limiting their usability outside specialized laboratories. This work introduces a code-free, large language model (LLM)-driven pipeline that converts natural language instructions into an end-to-end analysis. The system (1) inspects raw eye-tracking files to infer structure and metadata, (2) generates executable routines for data cleaning and detector implementation from concise user prompts, (3) applies the generated detector to label fixations and saccades, and (4) returns results and explanatory reports, and allows users to iteratively optimize their code by editing the prompt. Evaluated on public benchmarks, the approach achieves accuracy comparable to traditional methods while substantially reducing technical overhead. The framework lowers barriers to entry for eye-tracking research, providing a flexible and accessible alternative to code-intensive workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13241v1">Early-Warning Learner Satisfaction Forecasting in MOOCs via Temporal Event Transformers and LLM Text Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Learner satisfaction is a critical quality signal in massive open online courses (MOOCs), directly influencing retention, engagement, and platform reputation. Most existing methods infer satisfaction \emph{post hoc} from end-of-course reviews and star ratings, which are too late for effective intervention. In this paper, we study \textbf{early-warning satisfaction forecasting}: predicting a learner's eventual satisfaction score using only signals observed in the first $t$ days of a course (e.g., $t\!\in\!\{7, 14, 28\}$). We propose \textbf{TET-LLM}, a multi-modal fusion framework that combines (i) a \emph{temporal event Transformer} over fine-grained behavioral event sequences, (ii) \emph{LLM-based contextual embeddings} extracted from early textual traces such as forum posts and short feedback, and (iii) short-text \emph{topic/aspect distributions} to capture coarse satisfaction drivers. A heteroscedastic regression head outputs both a point estimate and a predictive uncertainty score, enabling conservative intervention policies. Comprehensive experiments on a large-scale multi-platform MOOC dataset demonstrate that TET-LLM consistently outperforms aggregate-feature and text-only baselines across all early-horizon settings, achieving an RMSE of 0.82 and AUC of 0.77 at the 7-day horizon. Ablation studies confirm the complementary contribution of each modality, and uncertainty calibration analysis shows near-nominal 90\% interval coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08125v3">Not All Tokens Matter: Towards Efficient LLM Reasoning via Token Significance in Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning abilities but often produce unnecessarily long explanations that reduce efficiency. Although reinforcement learning (RL) has been used to improve reasoning, most methods focus on accuracy and rely on uniform length-based rewards that overlook the differing contributions of individual tokens, often harming correctness. We revisit length optimization in RL through the perspective of token significance. Observing that many chain-of-thought (CoT) tokens contribute little to the final answer, we introduce a significance-aware length reward that selectively penalizes insignificance tokens, reducing redundancy while preserving essential reasoning. We also propose a dynamic length reward that encourages more detailed reasoning early in training and gradually shifts toward conciseness as learning progresses. Integrating these components into standard policy optimization yields a framework that improves both reasoning efficiency and accuracy. Experiments across multiple benchmarks demonstrate substantial reductions in response length while preserving or improving correctness, highlighting the importance of modeling token significance for efficient LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05096v2">RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) acquire most of their knowledge during pretraining, which ties them to a fixed snapshot of the world and makes adaptation to continuously evolving knowledge challenging. As facts, entities, and events change over time, models may experience continuous knowledge drift, resulting not only in outdated predictions but also in temporally inconsistent reasoning. Although existing approaches, such as continual finetuning, knowledge editing, and retrieval-augmented generation (RAG), aim to update or supplement model knowledge, they are rarely evaluated in settings that reflect chronological, evolving, and real-world knowledge evolution. In this work, we introduce a new benchmark of real-world dynamic events, constructed from time-stamped evidence that captures how knowledge evolves over time, which enables systematic evaluation of model adaptation under continuous knowledge drift. The benchmark reveals that most existing methods, including vanilla RAG and several learning-based approaches, struggle under this setting, exposing critical limitations such as catastrophic forgetting and temporal inconsistency. To mitigate these limitations, we propose a time-aware retrieval baseline, Chronos, which progressively organizes retrieved evidence into an Event Evolution Graph to enable more temporally consistent understanding in LLMs without additional training. Overall, this work provides a foundation for analyzing and advancing LLM adaptation to continuous knowledge drift in realistic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13226v1">KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rely heavily on Key-Value (KV) caching to minimize inference latency. However, standard KV caches are context-dependent: reusing a cached document in a new context requires recomputing KV states to account for shifts in attention distribution. Existing solutions such as CacheBlend, EPIC, and SAM-KV mitigate this issue by selectively recomputing a subset of tokens; however, they still incur non-negligible computational overhead (FLOPs) and increased Time-to-First-Token (TTFT) latency. In this paper, we propose KV Packet, a recomputation-free cache reuse framework that treats cached documents as immutable ``packets'' wrapped in light-weight trainable soft-token adapters, which are trained via self-supervised distillation to bridge context discontinuities. Experiments on Llama-3.1 and Qwen2.5 demonstrate that the proposed KV Packet method achieves near-zero FLOPs and lower TTFT than recomputation-based baselines, while retaining F1 scores comparable to those of the full recomputation baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09613v2">Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 duplicate of arXiv:2604.08075
    </div>
    <details class="paper-abstract">
      Production vLLM fleets provision every instance for worst-case context length, wasting 4-8x concurrency on the 80-95% of requests that are short and simultaneously triggering KV-cache failures -- OOM crashes, preemption storms, and request rejections. Both problems share a single root cause: configuration-traffic mismatch. We propose token-budget-aware pool routing: estimate each request's total token budget using a self-calibrating per-category bytes-per-token ratio, then dispatch it to one of two vLLM pools -- a high-throughput short pool or a high-capacity long pool -- each right-sized for its workload class. The ratio is learned online via exponential moving average from usage.prompt_tokens feedback, requiring no tokenizer. A closed-form cost model, savings = alpha * (1 - 1/rho), predicts fleet-level GPU savings from two observable quantities: the short-traffic fraction alpha and the throughput gain ratio rho. On traces from the Azure LLM Inference Dataset and LMSYS-Chat-1M serving Llama-3-70B on A100 GPUs, token-budget routing reduces GPU instances by 17-39% (\$1.2-2.0M/yr at 1,000 req/s), with savings verified by a self-contained discrete-event simulator. A case study projecting Qwen3-235B-A22B on AMD MI300X at 10,000 req/s shows \$15.4M/yr in savings. The algorithm adds O(1) dispatch overhead, self-calibrates across content types without a tokenizer, and composes with PagedAttention, continuous batching, and prefill-decode disaggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21440v4">KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to IJCNN 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: https://github.com/Wangshuaiia/KG-Hopper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12998v1">Personalizing LLM-Based Conversational Programming Assistants</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to the Doctoral and Early Career Symposium of the 19th International Conference on Cooperative and Human Aspects of Software Engineering (CHASE DECS 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown much promise in powering a variety of software engineering (SE) tools. Offering natural language as an intuitive interaction mechanism, LLMs have recently been employed as conversational ``programming assistants'' capable of supporting several SE activities simultaneously. As with any SE tool, it is crucial that these assistants effectively meet developers' needs. Recent studies have shown addressing this challenge is complicated by the variety in developers' needs, and the ambiguous and unbounded nature of conversational interaction. This paper discusses our current and future work towards characterizing how diversity in cognition and organizational context impacts developers' needs, and exploring personalization as a means of improving the inclusivity of LLM-based conversational programming assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11767v2">$λ_A$: A Typed Lambda Calculus for LLM Agent Composition</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Existing LLM agent frameworks lack formal semantics: there is no principled way to determine whether an agent configuration is well-formed or will terminate. We present $λ_A$, a typed lambda calculus for agent composition that extends the simply-typed lambda calculus with oracle calls, bounded fixpoints (the ReAct loop), probabilistic choice, and mutable environments. We prove type safety, termination of bounded fixpoints, and soundness of derived lint rules, with full Coq mechanization (1,519 lines, 42 theorems, 0 Admitted). As a practical application, we derive a lint tool that detects structural configuration errors directly from the operational semantics. An evaluation on 835 real-world GitHub agent configurations shows that 94.1% are structurally incomplete under $λ_A$, with YAML-only lint precision at 54%, rising to 96--100% under joint YAML+Python AST analysis on 175 samples. This gap quantifies, for the first time, the degree of semantic entanglement between declarative configuration and imperative code in the agent ecosystem. We further show that five mainstream paradigms (LangGraph, CrewAI, AutoGen, OpenAI SDK, Dify) embed as typed $λ_A$ fragments, establishing $λ_A$ as a unifying calculus for LLM agent composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13851v3">Evaluating LLM-Generated ACSL Annotations for Formal Verification</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 12 pages. Formal Techniques for Judicious Programming FTfJP-2026 at ECOOP. Conditionally Accepted. Final Revision
    </div>
    <details class="paper-abstract">
      Formal specifications are crucial for building verifiable and dependable software systems, yet generating accurate and verifiable specifications for real-world C programs remains challenging. This paper presents an empirical evaluation of automated ACSL annotation generation strategies for C programs, comparing a rule-based Python script, Frama-C's RTE plugin, and three large language models (DeepSeek-V3.2, GPT-5.2, and OLMo 3.1 32B Instruct). The study focuses on one-shot annotation generation, assessing how these approaches perform when directly applied to verification tasks. Using a filtered subset of the CASP benchmark, we evaluate generated annotations through Frama-C's WP plugin with multiple SMT solvers, analyzing proof success rates, solver timeouts, and internal processing time. Our results show that rule-based approaches remain more reliable for verification success, while LLM-based methods exhibit more variable performance. These findings highlight both the current limitations and the potential of LLMs as complementary tools for automated specification generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12948v1">Drawing on Memory: Dual-Trace Encoding Improves Cross-Session Recall in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 16 pages, 2 tables, 2 figures
    </div>
    <details class="paper-abstract">
      LLM agents with persistent memory store information as flat factual records, providing little context for temporal reasoning, change tracking, or cross-session aggregation. Inspired by the drawing effect [3], we introduce dual-trace memory encoding. In this method, each stored fact is paired with a concrete scene trace, a narrative reconstruction of the moment and context in which the information was learned. The agent is forced to commit to specific contextual details during encoding, creating richer, more distinctive memory traces. Using the LongMemEval-S benchmark (4,575 sessions, 100 recall questions), we compare dual-trace encoding against a fact-only control with matched coverage and format over 99 shared questions. Dual-trace achieves 73.7% overall accuracy versus 53.5%, a +20.2 percentage point (pp) gain (95% CI: [+12.1, +29.3], bootstrap p < 0.0001). Gains concentrate in temporal reasoning (+40pp), knowledge-update tracking (+25pp), and multi-session aggregation (+30pp), with no benefit for single-session retrieval, consistent with encoding specificity theory [8]. Token analysis shows dual-trace encoding achieves this gain at no additional cost. We additionally sketch an architectural design for adapting dual-trace encoding to coding agents, with preliminary pilot validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12944v1">Distorted or Fabricated? A Survey on Hallucination in Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 ACL 2026 findings
    </div>
    <details class="paper-abstract">
      Despite significant progress in video-language modeling, hallucinations remain a persistent challenge in Video Large Language Models (Vid-LLMs), referring to outputs that appear plausible yet contradict the content of the input video. This survey presents a comprehensive analysis of hallucinations in Vid-LLMs and introduces a systematic taxonomy that categorizes them into two core types: dynamic distortion and content fabrication, each comprising two subtypes with representative cases. Building on this taxonomy, we review recent advances in the evaluation and mitigation of hallucinations, covering key benchmarks, metrics, and intervention strategies. We further analyze the root causes of dynamic distortion and content fabrication, which often result from limited capacity for temporal representation and insufficient visual grounding. These insights inform several promising directions for future work, including the development of motion-aware visual encoders and the integration of counterfactual learning techniques. This survey consolidates scattered progress to foster a systematic understanding of hallucinations in Vid-LLMs, laying the groundwork for building robust and reliable video-language systems. An up-to-date curated list of related works is maintained at https://github.com/hukcc/Awesome-Video-Hallucination .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12913v1">CoDe-R: Refining Decompiler Output with LLMs via Rationale Guidance and Adaptive Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 10 pages, 7 figures, 6 tables. Accepted by IJCNN 2026
    </div>
    <details class="paper-abstract">
      Binary decompilation is a critical reverse engineering task aimed at reconstructing high-level source code from stripped executables. Although Large Language Models (LLMs) have recently shown promise, they often suffer from "logical hallucinations" and "semantic misalignment" due to the irreversible semantic loss during compilation, resulting in generated code that fails to re-execute. In this study, we propose Cognitive Decompiler Refinement with Robustness (CoDe-R), a lightweight two-stage code refinement framework. The first stage introduces Semantic Cognitive Enhancement (SCE), a Rationale-Guided Semantic Injection strategy that trains the model to recover high-level algorithmic intent alongside code. The second stage introduces a Dynamic Dual-Path Fallback (DDPF) mechanism during inference, which adaptively balances semantic recovery and syntactic stability via a hybrid verification strategy. Evaluation on the HumanEval-Decompile benchmark demonstrates that CoDe-R (using a 1.3B backbone) establishes a new State-of-the-Art (SOTA) in the lightweight regime. Notably, it is the first 1.3B model to exceed an Average Re-executability Rate of 50.00%, significantly outperforming the baseline and effectively bridging the gap between efficient models and expert-level performance. Our code is available at https://github.com/Theaoi/CoDe-R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10387v2">Leveraging Mathematical Reasoning of LLMs for Efficient GPU Thread Mapping</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 11 pages, 5 figures, 8 tables. Submitted to IEEE Transactions on Parallel and Distributed Systems (TPDS)
    </div>
    <details class="paper-abstract">
      Mapping parallel threads onto non-box-shaped domains is a known challenge in GPU computing; efficient mapping prevents performance penalties from unnecessary resource allocation. Currently, achieving this requires significant analytical human effort to manually derive bespoke mapping functions for each geometry. This work introduces a novel approach leveraging the symbolic reasoning of Large Language Models (LLMs) to automate this derivation entirely through in-context learning. Focusing on state-of-the-art open-weights models, we conducted a rigorous comparative analysis across spatial domains of increasing complexity. Our results demonstrate that modern local LLMs successfully infer exact O(1) and O(log N) mapping equations for complex 2D/3D dense domains and 2D fractals, vastly outperforming traditional symbolic regression methods. Crucially, we profile the energetic viability of this approach on high-performance infrastructure, distinguishing between the code-generation and execution phases. While one-time inference incurs a high energy penalty -- particularly for reasoning-focused models like DeepSeek-R1 -- this is a single upfront investment. Once integrated, the generated analytical kernels eliminate block waste entirely, yielding massive energy and time savings (e.g., up to 4833x speedup and 2890x energy reduction) during actual GPU workloads. Finally, we identify a current "reasoning ceiling" when these models face highly recursive 3D fractals (e.g., the Menger Sponge). This limitation benchmarks the present maturity of open-weight architectures, charting a viable path toward fully automated, energy-efficient GPU resource optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12881v1">Evaluating LLMs Code Reasoning Under Real-World Context</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted by ICES SRC (ACM Student Research Competition)
    </div>
    <details class="paper-abstract">
      Code reasoning tasks are increasingly crucial to evaluating large language models (LLMs). Yet most existing benchmarks rely on simplistic, LLM-generated snippets or human-written solutions to code challenges and often restrict inputs and outputs to primitive types, failing to reflect the structure and dependencies of real-world projects. These simplifications limit their ability to measure practical generalizability. We present R2Eval1, a benchmark of 135 code reasoning problems drawn from ten widely used Python projects. Unlike prior work, R2Eval serializes compound and custom types, preserving real-world data complexity and enabling a more realistic assessment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12851v1">Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Despite their global prevalence, many Large Language Models (LLMs) are aligned to a monolithic, often Western-centric set of values. This paper investigates the more challenging task of fine-grained value alignment: examining whether LLMs can emulate the distinct cultural values of demographic subgroups. Using Singapore as a case study and the World Values Survey (WVS), we examine the value landscape and show that even state-of-the-art models like GPT-4.1 achieve only 57.4% accuracy in predicting subgroup modal preferences. We construct a dataset of over 20,000 samples to train and evaluate a range of models. We demonstrate that simple fine-tuning on structured numerical preferences yields substantial gains, improving accuracy on unseen, out-of-distribution subgroups by an average of 17.4%. These gains partially transfer to open-ended generation. However, we find significant pre-existing performance biases, where models better emulate young, male, Chinese, and Christian personas. Furthermore, while fine-tuning improves average performance, it widens the disparity between subgroups when measured by distance-aware metrics. Our work offers insights into the limits and fairness implications of subgroup-level cultural alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09443v3">Many-Tier Instruction Hierarchy in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, other agents, and more-each carrying different levels of trust and authority. When these instructions conflict, agents must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12817v1">Understanding and Improving Continuous Adversarial Training for LLMs via In-context Learning Theory</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 The Fourteenth International Conference on Learning Representations (ICLR 2026)
    </div>
    <details class="paper-abstract">
      Adversarial training (AT) is an effective defense for large language models (LLMs) against jailbreak attacks, but performing AT on LLMs is costly. To improve the efficiency of AT for LLMs, recent studies propose continuous AT (CAT) that searches for adversarial inputs within the continuous embedding space of LLMs during AT. While CAT has achieved empirical success, its underlying mechanism, i.e., why adversarial perturbations in the embedding space can help LLMs defend against jailbreak prompts synthesized in the input token space, remains unknown. This paper presents the first theoretical analysis of CAT on LLMs based on in-context learning (ICL) theory. For linear transformers trained with adversarial examples from the embedding space on in-context linear regression tasks, we prove a robust generalization bound that has a negative correlation with the perturbation radius in the embedding space. This clearly explains why CAT can defend against jailbreak prompts from the LLM's token space. Further, the robust bound shows that the robustness of an adversarially trained LLM is closely related to the singular values of its embedding matrix. Based on this, we propose to improve LLM CAT by introducing an additional regularization term, which depends on singular values of the LLM's embedding matrix, into the objective function of CAT. Experiments on real-world LLMs demonstrate that our method can help LLMs achieve a better jailbreak robustness-utility tradeoff. The code is available at https://github.com/fshp971/continuous-adv-icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12816v1">The role of System 1 and System 2 semantic memory structure in human and LLM biases</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 31 pages, 5 figures, 9 appendix figures
    </div>
    <details class="paper-abstract">
      Implicit biases in both humans and large language models (LLMs) pose significant societal risks. Dual process theories propose that biases arise primarily from associative System 1 thinking, while deliberative System 2 thinking mitigates bias, but the cognitive mechanisms that give rise to this phenomenon remain poorly understood. To better understand what underlies this duality in humans, and possibly in LLMs, we model System 1 and System 2 thinking as semantic memory networks with distinct structures, built from comparable datasets generated by both humans and LLMs. We then investigate how these distinct semantic memory structures relate to implicit gender bias using network-based evaluation metrics. We find that semantic memory structures are irreducible only in humans, suggesting that LLMs lack certain types of human-like conceptual knowledge. Moreover, semantic memory structure relates consistently to implicit bias only in humans, with lower levels of bias in System~2 structures. These findings suggest that certain types of conceptual knowledge contribute to bias regulation in humans, but not in LLMs, highlighting fundamental differences between human and machine cognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09152v2">PrivacyReasoner: Can LLM Emulate a Human-like Privacy Mind?</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Prior work on LLM-based privacy focuses on norm judgment over synthetic vignettes, rather than how people think about a specific data practice and formulate their opinions. We address this gap by designing PrivacyReasoner, an agent architecture grounded in three key ideas: (1) LLMs can detect subtle privacy cues in natural language and role-play human characteristics; (2) a user's ``privacy mind'' can be reconstructed from their real-world online comment history, distilling experiences, personality, and cultural orientations; and (3) a contextual filter can dynamically activate relevant privacy beliefs based on the contexts in a scenario. We evaluate PrivacyReasoner on real-world privacy discussions from Hacker News, using an LLM-as-a-Judge evaluator calibrated against an established privacy concern taxonomy to quantify reasoning faithfulness. PrivacyReasoner significantly outperforms baselines in predicting individual privacy concerns and generalizes across different domains, such as AI, e-commerce, and healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.23728v3">GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to Findings of the Association for Computational Linguistics: ACL 2026
    </div>
    <details class="paper-abstract">
      With the increasing quality and spread of LLM assistants, the amount of generated content is growing rapidly. In many cases and tasks, such texts are already indistinguishable from those written by humans, and the quality of generation continues to increase. At the same time, detection methods are advancing more slowly than generation models, making it challenging to prevent misuse of generative AI technologies. We propose GigaCheck, a dual-strategy framework for AI-generated text detection. At the document level, we leverage the representation learning of fine-tuned LLMs to discern authorship with high data efficiency. At the span level, we introduce a novel structural adaptation that treats generated text segments as "objects." By integrating a DETR-like vision model with linguistic encoders, we achieve precise localization of AI intervals, effectively transferring the robustness of visual object detection to the textual domain. Experimental results across three classification and three localization benchmarks confirm the robustness of our approach. The shared fine-tuned backbone delivers strong accuracy in both scenarios, highlighting the generalization power of the learned embeddings. Moreover, we successfully demonstrate that visual detection architectures like DETR are not limited to pixel space, effectively generalizing to the localization of generated text spans. To ensure reproducibility and foster further research, we publicly release our source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12770v1">Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Editing human-written text has become a standard use case of large language models (LLMs), for example, to make one's arguments more appropriate for a discussion. Comparing human to LLM-generated edits, however, we observe a mismatch in editing strategies: While LLMs often perform multiple scattered edits and tend to change meaning notably, humans rather encapsulate dependent changes in self-contained, meaning-preserving edits. In this paper, we present a reinforcement learning approach that teaches LLMs human-like editing to improve the appropriateness of arguments. Our approach produces self-contained sentence-level edit suggestions that can be accepted or rejected independently. We train the approach using group relative policy optimization with a multi-component reward function that jointly optimizes edit-level semantic similarity, fluency, and pattern conformity as well as argument-level appropriateness. In automatic and human evaluation, it outperforms competitive baselines and the state of the art in human-like editing, with multi-round editing achieving appropriateness close to full rewriting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21990v4">ChemDFM-R: A Chemical Reasoning LLM Enhanced with Atomized Chemical Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 20 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Atomized chemical knowledge, such as functional group information of molecules and reactions, plays a pivotal intermediate role in the reasoning process that connects molecular structures with their properties and reactivities. While large language models (LLMs) have achieved impressive progress, the absence of atomized chemical knowledge results in their superficial understanding of chemistry and limited chemical reasoning capabilities. In this work, to tackle this problem, we develop a Chemical Reasoning LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized chemical knowledge, ChemFG, annotating the presence of functional groups in molecules and the changes of functional groups during chemical reactions, to enhance the model's understanding of the fundamental principles and internal logic of chemistry. Then, we propose a mixed-source distillation method that initializes the model's reasoning capability with limited distilled data, and develop a four-stage training pipeline to equip the model with atomized chemical knowledge and chemical reasoning logic. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs, surpassing both the general-domain LLMs and domain-specific chemical LLMs. Moreover, ChemDFM-R achieves comparable or superior performance compared with cutting-edge commercial LLMs, such as o4-mini. Further case studies illustrate how explicit reasoning chains significantly improve the model's reliability, transparency, and practicality in real-world human-AI collaboration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02830v2">GRADE: Probing Knowledge Gaps in LLMs through Gradient Subspace Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Detecting whether a model's internal knowledge is sufficient to correctly answer a given question is a fundamental challenge in deploying responsible LLMs. In addition to verbalising the confidence by LLM self-report, more recent methods explore the model internals, such as the hidden states of the response tokens, to capture how much knowledge is activated. We argue that such activated knowledge may not align with what the query requires, e.g., capturing the stylistic and length-related features that are uninformative for answering the query. To fill the gap, we propose GRADE (Gradient Dynamics for knowledge gap detection), which quantifies the knowledge gap via the cross-layer rank ratio of the gradient to that of the corresponding hidden state subspace. This is motivated by the property of gradients as estimators of the required knowledge updates for a given target. We validate GRADE on six benchmarks, demonstrating its effectiveness and robustness to input perturbations. In addition, we present a case study showing how the gradient chain can generate interpretable explanations of knowledge gaps for long-form answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12721v1">InsightFlow: LLM-Driven Synthesis of Patient Narratives for Mental Health into Causal Models</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Clinical case formulation organizes patient symptoms and psychosocial factors into causal models, often using the 5P framework. However, constructing such graphs from therapy transcripts is time consuming and varies across clinicians. We present InsightFlow, an LLM based approach that automatically generates 5P aligned causal graphs from patient-therapist dialogues. Using 46 psychotherapy intake transcripts annotated by clinical experts, we evaluate LLM generated graphs against human formulations using structural (NetSimile), semantic (embedding similarity), and expert rated clinical criteria. The generated graphs show structural similarity comparable to inter annotator agreement and high semantic alignment with human graphs. Expert evaluations rate the outputs as moderately complete, consistent, and clinically useful. While LLM graphs tend to form more interconnected structures compared to the chain like patterns of human graphs, overall complexity and content coverage are similar. These results suggest that LLMs can produce clinically meaningful case formulation graphs within the natural variability of expert practice. InsightFlow highlights the potential of automated causal modeling to augment clinical workflows, with future work needed to improve temporal reasoning and reduce redundancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05643v2">Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted by ACL2026 Findings
    </div>
    <details class="paper-abstract">
      Extending CoT through RL has been widely used to enhance the reasoning capabilities of LLMs. However, due to the sparsity of reward signals, it can also induce undesirable thinking patterns such as overthinking, i.e., generating redundant intermediate reasoning content. In this work, we argue that a major source of such redundancy is inefficient reflection, which often manifests in two problematic patterns: Indiscriminate Reflection, where the model performs broad, low-impact checks throughout reasoning, and Repetitive Reflection, where it repeatedly re-verifies an already established conclusion. To address this, we introduce a graph-based CoT optimization framework. Specifically, we convert each linear CoT into a directed acyclic graph (DAG) with explicit dependency edges, and design a dual pruning strategy: branch-level pruning removes weakly contributing reflection branches, while depth-level pruning eliminates late-stage re-verification. We distill this behavior via a three-stage pipeline: (1) SFT to initialize the policy on pruned concise traces, (2) DPO to prefer correct but less redundant trajectories, and (3) GRPO with length penalty to jointly optimize answer correctness and efficiency. Experiments show that our approach reduces the average reasoning tokens by 42\% while maintaining or improving accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12648v1">TimeSAF: Towards LLM-Guided Semantic Asynchronous Fusion for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Despite the recent success of large language models (LLMs) in time-series forecasting, most existing methods still adopt a Deep Synchronous Fusion strategy, where dense interactions between textual and temporal features are enforced at every layer of the network. This design overlooks the inherent granularity mismatch between modalities and leads to what we term semantic perceptual dissonance: high-level abstract semantics provided by the LLM become inappropriately entangled with the low-level, fine-grained numerical dynamics of time series, making it difficult for semantic priors to effectively guide forecasting. To address this issue, we propose TimeSAF, a new framework based on hierarchical asynchronous fusion. Unlike synchronous approaches, TimeSAF explicitly decouples unimodal feature learning from cross-modal interaction. It introduces an independent cross-modal semantic fusion trunk, which uses learnable queries to aggregate global semantics from the temporal and prompt backbones in a bottom-up manner, and a stage-wise semantic refinement decoder that asynchronously injects these high-level signals back into the temporal backbone. This mechanism provides stable and efficient semantic guidance while avoiding interference with low-level temporal dynamics. Extensive experiments on standard long-term forecasting benchmarks show that TimeSAF significantly outperforms state-of-the-art baselines, and further exhibits strong generalization in both few-shot and zero-shot transfer settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12640v1">LLMs Are Not a Silver Bullet: A Case Study on Software Fairness</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Fairness is a critical requirement for human-related, high-stakes software systems, motivating extensive research on bias mitigation. Prior work has largely focused on tabular data settings using traditional Machine Learning (ML) methods. With the rapid rise of Large Language Models (LLMs), recent studies have begun to explore their use for bias mitigation in the same setting. However, it remains unclear whether LLM-based methods offer advantages over traditional ML methods, leaving software engineers without clear guidance for practical adoption. To address this gap, we present a large-scale study comparing state-of-the-art ML- and LLM-based bias mitigation methods. We find that ML-based methods consistently outperform LLM-based methods in both fairness and predictive performance, with even strong LLMs failing to surpass established ML baselines. To understand why prior LLM-based studies report favorable results, we analyze their evaluation settings and show that these gains are largely driven by artificially balanced test data rather than realistic imbalanced distributions. We further observe that existing LLM-based methods primarily rely on in-context learning and thus fail to leverage all available training data. Motivated by this, we explore supervised fine-tuning on the full training set and find that, while it achieves competitive results, its advantages over traditional ML methods remain limited. These findings suggest that LLMs are not a silver bullet for software fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12634v1">RPRA: Predicting an LLM-Judge for Efficient but Performant Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 10 pages in main text + 6 pages of references + 36 pages of appendices, 12 figures in main text + 37 figures in appendices, 2 tables in main text + 3 table in appendices, 13 prompts in appendices
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans and have models ask for help when they believe they are incapable of solving a problem on their own; we can overcome this trade-off by allowing smaller models to respond to queries when they believe they can provide good responses, and deferring to larger models when they do not believe they can. To this end, in this paper, we investigate the viability of Predict-Answer/Act (PA) and Reason-Predict-Reason-Answer/Act (RPRA) paradigms where models predict -- prior to responding -- how an LLM judge would score their output. We evaluate three approaches: zero-shot prediction, prediction using an in-context report card, and supervised fine-tuning. Our results show that larger models (particularly reasoning models) perform well when predicting generic LLM judges zero-shot, while smaller models can reliably predict such judges well after being fine-tuned or provided with an in-context report card. Altogether, both approaches can substantially improve the prediction accuracy of smaller models, with report cards and fine-tuning achieving mean improvements of up to 55% and 52% across datasets, respectively. These findings suggest that models can learn to predict their own performance limitations, paving the way for more efficient and self-aware AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12632v1">Calibration-Aware Policy Optimization for Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Published as a conference paper at ACL 2026
    </div>
    <details class="paper-abstract">
      Group Relative Policy Optimization (GRPO) enhances LLM reasoning but often induces overconfidence, where incorrect responses yield lower perplexity than correct ones, degrading relative calibration as described by the Area Under the Curve (AUC). Existing approaches either yield limited improvements in calibration or sacrifice gains in reasoning accuracy. We first prove that this degradation in GRPO-style algorithms stems from their uncertainty-agnostic advantage estimation, which inevitably misaligns optimization gradients with calibration. This leads to improved accuracy at the expense of degraded calibration. We then propose Calibration-Aware Policy Optimization (CAPO). It adopts a logistic AUC surrogate loss that is theoretically consistent and admits regret bound, enabling uncertainty-aware advantage estimation. By further incorporating a noise masking mechanism, CAPO achieves stable learning dynamics that jointly optimize calibration and accuracy. Experiments on multiple mathematical reasoning benchmarks show that CAPO-1.5B significantly improves calibration by up to 15% while achieving accuracy comparable to or better than GRPO, and further boosts accuracy on downstream inference-time scaling tasks by up to 5%. Moreover, when allowed to abstain under low-confidence conditions, CAPO achieves a Pareto-optimal precision-coverage trade-off, highlighting its practical value for hallucination mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22359v4">League of LLMs: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have shown exceptional capabilities across a wide range of tasks, reliable evaluation remains a critical challenge due to data contamination, opaque operation, and subjective preferences. To address these issues, we propose League of LLMs (LOL), a novel benchmark-free evaluation paradigm that organizes multiple LLMs into a self-governed league for multi-round mutual evaluation. LOL integrates four core criteria (dynamic, transparent, objective, and professional) to mitigate key limitations of existing paradigms. Experiments on eight mainstream LLMs in mathematics and programming demonstrate that LOL can effectively distinguish LLM capabilities while maintaining high internal ranking stability (Top-$k$ consistency $= 70.7\%$). Beyond ranking, LOL reveals empirical findings that are difficult for traditional paradigms to capture. For instance, ``memorization-based answering'' behaviors are observed in some models, and higher in-family scores are found in the OpenAI model family ($Δ= 9$, $p < 0.05$). Finally, we make our framework and code publicly available as a valuable complement to the current LLM evaluation ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12615v1">DeepTest Tool Competition 2026: Benchmarking an LLM-Based Automotive Assistant</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Published in the proceedings of the DeepTest workshop at the 48th International Conference on Software Engineering (ICSE) 2026
    </div>
    <details class="paper-abstract">
      This report summarizes the results of the first edition of the Large Language Model (LLM) Testing competition, held as part of the DeepTest workshop at ICSE 2026. Four tools competed in benchmarking an LLM-based car manual information retrieval application, with the objective of identifying user inputs for which the system fails to appropriately mention warnings contained in the manual. The testing solutions were evaluated based on their effectiveness in exposing failures and the diversity of the discovered failure-revealing tests. We report on the experimental methodology, the competitors, and the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12610v1">Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) mitigates hallucination in large language models (LLMs) by incorporating external knowledge during generation. However, the effectiveness of RAG depends not only on the design of the retriever and the capacity of the underlying model, but also on how retrieved evidence is structured and aligned with the query. Existing RAG approaches typically retrieve and concatenate unstructured text fragments as context, which often introduces redundant or weakly relevant information. This practice leads to excessive context accumulation, reduced semantic alignment, and fragmented reasoning chains, thereby degrading generation quality while increasing token consumption. To address these challenges, we propose Tri-RAG, a structured triplet-based retrieval framework that improves retrieval efficiency through reasoning-aligned context construction. Tri-RAG automatically transforms external knowledge from natural language into standardized structured triplets consisting of Condition, Proof, and Conclusion, explicitly capturing logical relations among knowledge fragments using lightweight prompt-based adaptation with frozen model parameters. Building on this representation, the triplet head Condition is treated as an explicit semantic anchor for retrieval and matching, enabling precise identification of query-relevant knowledge units without directly concatenating lengthy raw texts. As a result, Tri-RAG achieves a favorable balance between retrieval accuracy and context token efficiency. Experimental results across multiple benchmark datasets demonstrate that Tri-RAG significantly improves retrieval quality and reasoning efficiency, while producing more stable generation behavior and more efficient resource utilization in complex reasoning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12601v1">LLM-Guided Prompt Evolution for Password Guessing</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Passwords still remain a dominant authentication method, yet their security is routinely subverted by predictable user choices and large-scale credential leaks. Automated password guessing is a key tool for stress-testing password policies and modeling attacker behavior. This paper applies LLM-driven evolutionary computation to automatically optimize prompts for the LLM password guessing framework. Using OpenEvolve, an open-source system combining MAP-Elites quality-diversity search with an island population model we evolve prompts that maximize cracking rate on a RockYou-derived test set. We evaluate three configurations: a local setup with Qwen3 8B, a single compact cloud model Gemini-2.5 Flash, and a two-model ensemble of frontier LLMs. The approach raises the cracking rates from 2.02\% to 8.48\%. Character distribution analysis further confirms how evolved prompts produce statistically more realistic passwords. Automated prompt evolution is a low-barrier yet effective way to strengthen LLM-based password auditing and underlining how attack pipelines show tendency via automated improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12573v1">IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly deployed for decision-making, yet their adoption in high-stakes domains remains limited by miscalibrated probabilities, unfaithful explanations, and inability to incorporate expert knowledge precisely. We propose IDEA, a framework that extracts LLM decision knowledge into an interpretable parametric model over semantically meaningful factors. Through joint learning of verbal-to-numerical mappings and decision parameters via EM, correlated sampling that preserves factor dependencies, and direct parameter editing with mathematical guarantees, IDEA produces calibrated probabilities while enabling quantitative human-AI collaboration. Experiments across five datasets show IDEA with Qwen-3-32B (78.6%) outperforms DeepSeek R1 (68.1%) and GPT-5.2 (77.9%), achieving perfect factor exclusion and exact calibration -- precision unattainable through prompting alone. The implementation is publicly available at https://github.com/leonbig/IDEA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14732v2">INFORM-CT: INtegrating LLMs and VLMs FOR Incidental Findings Management in Abdominal CT</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted for Spotlight presentation at MIDL 2026
    </div>
    <details class="paper-abstract">
      Incidental findings in CT scans, though often benign, can have significant clinical implications and should be reported following established guidelines. Traditional manual inspection by radiologists is time-consuming and variable. This paper proposes a novel framework that leverages large language models (LLMs) and foundational vision-language models (VLMs) in a plan-and-execute agentic approach to improve the efficiency and precision of incidental findings detection, classification, and reporting for abdominal CT scans. Given medical guidelines for abdominal organs, the process of managing incidental findings is automated through a planner-executor framework. The planner, based on LLM, generates Python scripts using predefined base functions, while the executor runs these scripts to perform the necessary checks and detections, via VLMs, segmentation models, and image processing subroutines. We demonstrate the effectiveness of our approach through experiments on a CT abdominal benchmark for three organs, in a fully automatic end-to-end manner. Our results show that the proposed framework outperforms existing pure VLM-based approaches in terms of accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06846v2">CKG-LLM: LLM-Assisted Detection of Smart Contract Access Control Vulnerabilities Based on Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 6 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Traditional approaches for smart contract analysis often rely on intermediate representations such as abstract syntax trees, control-flow graphs, or static single assignment form. However, these methods face limitations in capturing both semantic structures and control logic. Knowledge graphs, by contrast, offer a structured representation of entities and relations, enabling richer intermediate abstractions of contract code and supporting the use of graph query languages to identify rule-violating elements. This paper presents CKG-LLM, a framework for detecting access-control vulnerabilities in smart contracts. Leveraging the reasoning and code generation capabilities of large language models, CKG-LLM translates natural-language vulnerability patterns into executable queries over contract knowledge graphs to automatically locate vulnerable code elements. Experimental evaluation demonstrates that CKG-LLM achieves superior performance in detecting access-control vulnerabilities compared to existing tools. Finally, we discuss potential extensions of CKG-LLM as part of future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12545v1">Cross-Cultural Simulation of Citizen Emotional Responses to Bureaucratic Red Tape Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 To appear in the CHI 2026 Workshop on PoliSim
    </div>
    <details class="paper-abstract">
      Improving policymaking is a central concern in public administration. Prior human subject studies reveal substantial cross-cultural differences in citizens' emotional responses to red tape during policy implementation. While LLM agents offer opportunities to simulate human-like responses and reduce experimental costs, their ability to generate culturally appropriate emotional responses to red tape remains unverified. To address this gap, we propose an evaluation framework for assessing LLMs' emotional responses to red tape across diverse cultural contexts. As a pilot study, we apply this framework to a single red-tape scenario. Our results show that all models exhibit limited alignment with human emotional responses, with notably weaker performance in Eastern cultures. Cultural prompting strategies prove largely ineffective in improving alignment. We further introduce \textbf{RAMO}, an interactive interface for simulating citizens' emotional responses to red tape and for collecting human data to improve models. The interface is publicly available at https://ramo-chi.ivia.ch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12543v1">A Two-Stage LLM Framework for Accessible and Verified XAI Explanations</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 8 pages, 8 figures, Accepted for publication at the 2026 IEEE World Congress on Computational Intelligence (WCCI 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to translate the technical outputs of eXplainable Artificial Intelligence (XAI) methods into accessible natural-language explanations. However, existing approaches often lack guarantees of accuracy, faithfulness, and completeness. At the same time, current efforts to evaluate such narratives remain largely subjective or confined to post-hoc scoring, offering no safeguards to prevent flawed explanations from reaching end-users. To address these limitations, this paper proposes a Two-Stage LLM Meta-Verification Framework that consists of (i) an Explainer LLM that converts raw XAI outputs into natural-language narratives, (ii) a Verifier LLM that assesses them in terms of faithfulness, coherence, completeness, and hallucination risk, and (iii) an iterative refeed mechanism that uses the Verifier's feedback to refine and improve them. Experiments across five XAI techniques and datasets, using three families of open-weight LLMs, show that verification is crucial for filtering unreliable explanations while improving linguistic accessibility compared with raw XAI outputs. In addition, the analysis of the Entropy Production Rate (EPR) during the refinement process indicates that the Verifier's feedback progressively guides the Explainer toward more stable and coherent reasoning. Overall, the proposed framework provides an efficient pathway toward more trustworthy and democratized XAI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12540v1">When Does Data Augmentation Help? Evaluating LLM and Back-Translation Methods for Hausa and Fongbe NLP</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 13 pages, 6 tables; previously submitted to KDD 2026
    </div>
    <details class="paper-abstract">
      Data scarcity limits NLP development for low-resource African languages. We evaluate two data augmentation methods -- LLM-based generation (Gemini 2.5 Flash) and back-translation (NLLB-200) -- for Hausa and Fongbe, two West African languages that differ substantially in LLM generation quality. We assess augmentation on named entity recognition (NER) and part-of-speech (POS) tagging using MasakhaNER 2.0 and MasakhaPOS benchmarks. Our results reveal that augmentation effectiveness depends on task type rather than language or LLM quality alone. For NER, neither method improves over baseline for either language; LLM augmentation reduces Hausa NER by 0.24% F1 and Fongbe NER by 1.81% F1. For POS tagging, LLM augmentation improves Fongbe by 0.33% accuracy, while back-translation improves Hausa by 0.17%; back-translation reduces Fongbe POS by 0.35% and has negligible effect on Hausa POS. The same LLM-generated synthetic data produces opposite effects across tasks for Fongbe -- hurting NER while helping POS -- suggesting task structure governs augmentation outcomes more than synthetic data quality. These findings challenge the assumption that LLM generation quality predicts augmentation success, and provide actionable guidance: data augmentation should be treated as a task-specific intervention rather than a universally beneficial preprocessing step.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16615v2">LLM-Guided Task- and Affordance-Level Exploration in Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 8 pages, 7 figures, ICRA 2026
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a promising approach for robotic manipulation, but it can suffer from low sample efficiency and requires extensive exploration of large state-action spaces. Recent methods leverage the commonsense knowledge and reasoning abilities of large language models (LLMs) to guide exploration toward more meaningful states. However, LLMs can produce plans that are semantically plausible yet physically infeasible, yielding unreliable behavior. We introduce LLM-TALE, a framework that uses LLMs' planning to directly steer RL exploration. LLM-TALE integrates planning at both the task level and the affordance level, improving learning efficiency by directing agents toward semantically meaningful actions. Unlike prior approaches that assume optimal LLM-generated plans or rewards, LLM-TALE corrects suboptimality online and explores multimodal affordance-level plans without human supervision. We evaluate LLM-TALE on pick-and-place tasks in standard RL benchmarks, observing improvements in both sample efficiency and success rates over strong baselines. Real-robot experiments indicate promising zero-shot sim-to-real transfer. Code and supplementary material are available at llm-tale.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12497v1">Adaptive Budget Allocation in LLM-Augmented Surveys</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate survey responses at low cost, but their reliability varies substantially across questions and is unknown before data collection. Deploying LLMs in surveys still requires costly human responses for verification and correction. How should a limited human-labeling budget be allocated across questions in real time? We propose an adaptive allocation algorithm that learns which questions are hardest for the LLM while simultaneously collecting human responses. Each human label serves a dual role: it improves the estimate for that question and reveals how well the LLM predicts human responses on it. The algorithm directs more budget to questions where the LLM is least reliable, without requiring any prior knowledge of question-level LLM accuracy. We prove that the allocation gap relative to the best possible allocation vanishes as the budget grows, and validate the approach on both synthetic data and a real survey dataset with 68 questions and over 2000 respondents. On real survey data, the standard practice of allocating human labels uniformly across questions wastes 10--12% of the budget relative to the optimal; our algorithm reduces this waste to 2--6%, and the advantage grows as questions become more heterogeneous in LLM prediction quality. The algorithm achieves the same estimation quality as traditional uniform sampling with fewer human samples, requires no pilot study, and is backed by formal performance guarantees validated on real survey data. More broadly, the framework applies whenever scarce human oversight must be allocated across tasks where LLM reliability is unknown.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14264v3">AAPO: Enhancing the Reasoning Capabilities of LLMs with Advantage Margin</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to ACL2026 Main Conference
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a margin-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO. Code is available at https://github.com/JianxXiong/AAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12469v1">Analyzing the Effect of Noise in LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Fine-tuning is the dominant paradigm for adapting pretrained large language models (LLMs) to downstream NLP tasks. In practice, fine-tuning datasets may contain various forms of noise arising from annotation errors, preprocessing artifacts, or automated data collection. While prior work has focused on designing robust learning algorithms to mitigate performance degradation under noisy conditions, comparatively little is known about how different types of noise affect the internal learning dynamics of LLMs during fine-tuning. In this work, we systematically study the impact of noise on model behavior across three pretrained model families (GPT-2, Qwen2 and Llama-2) and three diverse NLP tasks. We introduce controlled perturbations corresponding to three common real-world noise types: label noise, grammatical noise, and typographical noise. Beyond task-level performance, we analyze layer-wise representation changes and attention patterns to understand how noise propagates through the network. Our results show that corrupting labels (i.e. label noise) consistently causes the largest performance degradation, whereas grammatical noise and typographical noise can occasionally yield mild regularization benefits. We further find that noise effects are localized primarily to task-specific layers, while attention structures remain comparatively stable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12461v1">CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 ACL 2026, Main
    </div>
    <details class="paper-abstract">
      LLM-based Multi-Agent Systems (MAS) have demonstrated remarkable capabilities in solving complex tasks. Central to MAS is the communication topology which governs how agents exchange information internally. Consequently, the security of communication topologies has attracted increasing attention. In this paper, we investigate a critical privacy risk: MAS communication topologies can be inferred under a restrictive black-box setting, exposing system vulnerabilities and posing significant intellectual property threats. To explore this risk, we propose Communication Inference Attack (CIA), a novel attack that constructs new adversarial queries to induce intermediate agents' reasoning outputs and models their semantic correlations through the proposed global bias disentanglement and LLM-guided weak supervision. Extensive experiments on MAS with optimized communication topologies demonstrate the effectiveness of CIA, achieving an average AUC of 0.87 and a peak AUC of up to 0.99, thereby revealing the substantial privacy risk in MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12459v1">Operationalising the Right to be Forgotten in LLMs: A Lightweight Sequential Unlearning Framework for Privacy-Aligned Deployment in Politically Sensitive Environments</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in politically sensitive environments, where memorisation of personal data or confidential content raises regulatory concerns under frameworks such as the GDPR and its Right to be Forgotten. Translating such legal principles into large-scale generative systems presents significant technical challenges. We introduce a lightweight sequential unlearning framework that explicitly separates retention and suppression objectives. The method first stabilises benign capabilities through positive fine-tuning, then applies layer-restricted negative fine-tuning to suppress designated sensitive patterns while preserving general language competence. Experiments on the SemEval-2025 LLM Unlearning benchmark demonstrate effective behavioural suppression with minimal impact on factual accuracy and fluency. GPT-2 exhibits greater robustness than DistilGPT-2, highlighting the role of model capacity in privacy-aligned adaptation. We position sequential unlearning as a practical and reproducible mechanism for operationalising data erasure requirements in politically deployed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06443v2">Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 MobiSys 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on edge devices. To meet strict resource constraints, real-world deployment has pushed LLM quantization from 8-bit to 4-bit, 2-bit, and now 1.58-bit. Combined with lookup table (LUT)-based inference, CPUs run these ultra-low-bit LLMs even faster than NPUs, opening new opportunities for ubiquitous on-device intelligence. However, this paper identifies that LUT-based inference underutilizes memory bandwidth during parallel inference, which is required for prefilling, test-time scaling, and other multi-token scenarios. The root cause is the scalar LUT paradigm, which performs repetitive and non-contiguous memory accesses for each token. To solve the issue, we propose vector LUT, a new lookup paradigm that constructs a unified LUT across parallel tokens, and performs a single $1 \rightarrow N$ lookup per index. To realize it efficiently, we further introduce (1) Vector LUT-Centric Tensor Layout, and (2) Cache-Aware Streamed Lookup techniques. Evaluations on 5 edge devices across 3 LLMs show that Vec-LUT outperforms state-of-the-art baselines by up to $4.2\times$. Our implementation is integrated into llama.cpp. The code is available at https://github.com/OpenBitSys/vlut.cpp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12401v1">Three Birds, One Stone: Solving the Communication-Memory-Privacy Trilemma in LLM Fine-tuning Over Wireless Networks with Zeroth-Order Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Federated Learning (FL) offers a promising pathway for collaboratively fine-tuning Large Language Models (LLMs) at the edge; however, this paradigm faces a critical bottleneck: the prohibitive communication and memory overheads incurred by exchanging high-dimensional gradients. Furthermore, recent studies reveal that user training data can still be recovered from these local gradients, undermining the core privacy promise of FL. In this paper, we address this trilemma of communication, memory, and privacy by proposing pAirZero, a novel framework that synergizes Zeroth-Order (ZO) optimization with Over-the-Air (OTA) computation. Uniquely, pAirZero enables resource-constrained devices to submit their local gradient with only bit-level communication loads while participating in federated fine-tuning of LLMs with inference-level memory costs. This approach not only eliminates the high memory requirements needed for LLM fine-tuning but also alleviates the strict synchronization requirements that plague conventional OTA methods. We further formulate a rigorous optimization model to adaptively determine the optimal transmit power and noise levels, ensuring consistent privacy protection regardless of channel conditions. Numerical experiments demonstrate the superiority of pAirZero in enabling secure, efficient LLM fine-tuning over wireless networks, with only 25% peak memory cost on OPT-125M and communication load orders of magnitude lower than conventional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12385v1">From Myopic Selection to Long-Horizon Awareness: Sequential LLM Routing for Multi-Turn Dialogue</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Multi-turn dialogue is the predominant form of interaction with large language models (LLMs). While LLM routing is effective in single-turn settings, existing methods fail to maximize cumulative performance in multi-turn dialogue due to interaction dynamics and delayed rewards. To address this challenge, we move from myopic, single-turn selection to long-horizon sequential routing for multi-turn dialogue. Accordingly, we propose DialRouter, which first performs MCTS to explore dialogue branches induced by different LLM selections and collect trajectories with high cumulative rewards. DialRouter then learns a lightweight routing policy from search-derived data, augmented with retrieval-based future state approximation, enabling multi-turn routing without online search. Experiments on both open-domain and domain-specific dialogue tasks across diverse candidate sets of both open-source and closed-source LLMs demonstrate that DialRouter significantly outperforms single LLMs and existing routing baselines in task success rate, while achieving a superior performance-cost trade-off when combined with a cost-aware reward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12378v1">ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Despite advances in multilingual capabilities, most large language models (LLMs) remain English-centric in their training and, crucially, in their production of reasoning traces. Even when tasked with non-English problems, these models predominantly reason in English, creating a fundamental mismatch for non-English usage scenarios. We address this disparity directly with three contributions. (i) We introduce ReasonXL, the first large-scale parallel corpus of cross-domain reasoning traces spanning five European languages (English, German, French, Italian, and Spanish), with over two million aligned samples per language, each comprising prompts, reasoning traces, and final outputs, enabling direct supervision of language-specific reasoning. (ii) Using ReasonXL, we demonstrate that LLMs can be adapted to reason entirely in a desired target language, using a simple two-stage pipeline of supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR). The resulting models match or exceed baseline performance, with minimal loss in general knowledge and broadly preserved cross-lingual transfer. (iii) We conduct an extensive representational analysis of the adaptation and find a clear functional division across model depth: early layers contain an activation bottleneck that causally determines language identity, while upper layers concentrate the weight and activation changes driven by adaptation. We further find that RLVR achieves greater behavioral divergence from the base model with smaller parameter updates than SFT, suggesting a more efficient representational rerouting despite much smaller weight updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12376v1">Cooperative Memory Paging with Keyword Bookmarks for Long-Horizon LLM Conversations</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 16 pages, 10 figures, 16 tables
    </div>
    <details class="paper-abstract">
      When LLM conversations grow beyond the context window, old content must be evicted -- but how does the model recover it when needed? We propose cooperative paging: evicted segments are replaced with minimal keyword bookmarks ([pN:keywords], ~8-24 tokens each), and the model is given a recall() tool to retrieve full content on demand. On the LoCoMo benchmark (10 real multi-session conversations, 300+ turns), cooperative paging achieves the highest answer quality among six methods -- outperforming truncation, BM25, word-overlap retrieval, a search-tool baseline, and full context -- on four models (GPT-4o-mini, DeepSeek-v3.2, Claude Haiku, GLM-5), confirmed by four independent LLM judges ($p=0.017$, paired bootstrap). We then study the paging design space with a 5x4 ablation over boundary strategies and eviction policies (3,176 synthetic probes, 1,600 LoCoMo probes). Key findings: (1) coarse fixed-size pages (fixed_20) reach 96.7% while content-aware topic_shift collapses to 56.7%; (2) eviction policy choice is data-dependent (FIFO best on synthetic, LFU on LoCoMo); (3) two bookmark generation strategies improve over the heuristic baseline (+4.4 and +8.7 E2E points); (4) the remaining bottleneck is bookmark discrimination -- the model triggers recall() 96% of the time but selects the correct page only 57% when bookmarks are insufficiently distinctive. Keyword specificity alone accounts for a 25 percentage point accuracy difference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03174v2">LLM as Attention-Informed NTM and Topic Modeling as long-input Generation: Interpretability and long-Context Capability</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Topic modeling aims to produce interpretable topic representations and topic--document correspondences from corpora, but classical neural topic models (NTMs) remain constrained by limited representation assumptions and semantic abstraction ability. We study LLM-based topic modeling from both white-box and black-box perspectives. For white-box LLMs, we propose an attention-informed framework that recovers interpretable structures analogous to those in NTMs, including document-topic and topic-word distributions. This validates the view that LLM can serve as an attention-informed NTM. For black-box LLMs, we reformulate topic modeling as a structured long-input task and introduce a post-generation signal compensation method based on diversified topic cues and hybrid retrieval. Experiments show that recovered attention structures support effective topic assignment and keyword extraction, while black-box long-context LLMs achieve competitive or stronger performance than other baselines. These findings suggest a connection between LLMs and NTMs and highlight the promise of long-context LLMs for topic modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06928v2">Leveraging LLMs and Heterogeneous Knowledge Graphs for Persona-Driven Session-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Session-based recommendation systems (SBRS) aim to capture user's short-term intent from interaction sequences. However, the common assumption of anonymous sessions limits personalization, particularly under sparse or cold-start conditions. Recent advances in LLM augmented recommendation have shown that LLMs can generate rich item representations, but modeling user personas with LLMs remains challenging due to anonymous sessions. In this work, we propose a persona driven SBRS framework that explicitly models latent user personas inferred from a heterogeneous knowledge graph (KG) and integrates them into a data-driven SBRS. Our framework adopts a two-stage architecture consisting of personalized information extraction and personalized information utilization. In the personalized information extraction stage, we construct a heterogeneous KG that integrates time-independent user-item interactions, item-item relations, item-feature associations, and external metadata from DBpedia. We then learn latent user personas in an unsupervised manner using a Heterogeneous Deep Graph Infomax (HDGI) objective over a KG initialized with LLM-derived item embeddings. In the personalized information utilization stage, the learned persona representations together with LLM-derived item embeddings are incorporated into a modified architecture of data-driven SBRS to generate a candidate set of relevant items, followed by reranking using the base sequential model to emphasize short-term session intent. Unlike prior approaches that rely solely on sequence modeling or text-based user representations, our method grounds user persona modeling in structured relational signals derived from a heterogeneous KG. Experiments on Amazon Books and Amazon Movies & TV demonstrate that our approach consistently improves over sequential models with user embeddings derived using session history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12329v1">UniDetect: LLM-Driven Universal Fraud Detection across Heterogeneous Blockchains</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      As cross-chain interoperability advances, decentralized finance (DeFi) protocols enable illicit funds to be reorganized into uniform liquid assets that flow throughout the cryptocurrency market. Such operations can bypass monitoring targeted at individual blockchains and thereby weaken current regulatory frameworks. Motivated by these, we introduce UniDetect, a multi-chain cryptocurrency fraud account detection method based on large language models (LLMs). Specifically, we use domain knowledge to guide the LLM to generate general transaction summary texts applicable to heterogeneous blockchain accounts, which serve as evidence for fraud account detection. Furthermore, we introduce a two-stage alternating training strategy to continuously and dynamically enhance the multimodal joint reasoning for detecting fraudulent accounts based on both the textual evidence and the transaction graph patterns. Experiments on multiple blockchains show that UniDetect outperforms existing methods 5.57% to 7.58% in Kolmogorov-Smirnov (KS). For cross-chain zero-shot detection, UniDetect identifies over 94.58% of fraudulent accounts. It also generalizes well to non-blockchain data, delivering a 6.06% improvement in F1 over existing methods. The dataset and source code are available at https://github.com/msy0513/UniDetect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12312v1">CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as task-oriented agents in enterprise environments, ensuring their strict adherence to complex, domain-specific operational guidelines is critical. While utilizing an LLM-as-a-Judge is a promising solution for scalable evaluation, the reliability of these judges in detecting specific policy violations remains largely unexplored. This gap is primarily due to the lack of a systematic data generation method, which has been hindered by the extensive cost of fine-grained human annotation and the difficulty of synthesizing realistic agent violations. In this paper, we introduce CompliBench, a novel benchmark designed to evaluate the ability of LLM judges to detect and localize guideline violations in multi-turn dialogues. To overcome data scarcity, we develop a scalable, automated data generation pipeline that simulates user-agent interactions. Our controllable flaw injection process automatically yields precise ground-truth labels for the violated guideline and the exact conversation turn, while an adversarial search method ensures these introduced perturbations are highly challenging. Our comprehensive evaluation reveals that current state-of-the-art proprietary LLMs struggle significantly with this task. In addition, we demonstrate that a small-scale judge model fine-tuned on our synthesized data outperforms leading LLMs and generalizes well to unseen business domains, highlighting our pipeline as an effective foundation for training robust generative reward models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12311v1">Is Vibe Coding the Future? An Empirical Assessment of LLM Generated Codes for Construction Safety</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      The emergence of vibe coding, a paradigm where non-technical users instruct Large Language Models (LLMs) to generate executable codes via natural language, presents both significant opportunities and severe risks for the construction industry. While empowering construction personnel such as the safety managers, foremen, and workers to develop tools and software, the probabilistic nature of LLMs introduces the threat of silent failures, wherein generated code compiles perfectly but executes flawed mathematical safety logic. This study empirically evaluates the reliability, software architecture, and domain-specific safety fidelity of 450 vibe-coded Python scripts generated by three frontier models, Claude 3.5 Haiku, GPT-4o-Mini, and Gemini 2.5 Flash. Utilizing a persona-driven prompt dataset (n=150) and a bifurcated evaluation pipeline comprising isolated dynamic sandboxing and an LLM-as-a-Judge, the research quantifies the severe limits of zero-shot vibe codes for construction safety. The findings reveal a highly significant relationship between user persona and data hallucination, demonstrating that less formal prompts drastically increase the AI's propensity to invent missing safety variables. Furthermore, while the models demonstrated high foundational execution viability (~85%), this syntactic reliability actively masked logic deficits and a severe lack of defensive programming. Among successfully executed scripts, the study identified an alarming ~45% overall Silent Failure Rate, with GPT-4o-Mini generating mathematically inaccurate outputs in ~56% of its functional code. The results demonstrate that current LLMs lack the deterministic rigor required for standalone safety engineering, necessitating the adoption of deterministic AI wrappers and strict governance for cyber-physical deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09601v2">Hubble: An LLM-Driven Agentic Framework for Safe, Diverse, and Reproducible Alpha Factor Discovery</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Automated alpha discovery is difficult because the search space of formulaic factors is combinatorial, the signal-to-noise ratio in daily equity data is low, and unconstrained program generation is operationally unsafe. We present Hubble, an agentic factor mining framework that combines large language models (LLMs) with a domain-specific operator language, an abstract syntax tree (AST) execution sandbox, a dual-channel retrieval-augmented generation (RAG) module, and a family-aware selection mechanism. Instead of treating the LLM as an unconstrained code generator, Hubble restricts generation to interpretable operator trees, evaluates every candidate through a deterministic cross-sectional pipeline, and feeds back both top formulas and structured family-level diagnostics to subsequent rounds. The current system additionally introduces positive/negative RAG, formula-similarity penalties, standardized multi-metric scoring, dual reporting of RankIC and Pearson IC, and persistent diagnostics artifacts for post-hoc research analysis. On a U.S. equity universe of roughly 500 stocks, our main run evaluates 104 valid candidates across three rounds with zero runtime crashes and discovers a top set dominated by range, volatility, and trend families rather than crowded volume-only motifs. We then fix the resulting top-5 factors and validate them on a held-out period from 2025-06-01 to 2026-03-13. In this out-of-sample window, the two range factors and two volatility factors remain positive and several achieve HAC-significant Pearson IC and long-short evidence, whereas the weakest in-sample trend factor decays materially. These results suggest that safe LLM-guided search can be upgraded from a syntax-compliant generator into a reproducible alpha-research workflow that jointly optimizes validity, diversity, interpretability, and family-level generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02546v2">To trust or not to trust: Attention-based Trust Management for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to ACL 2026 main
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have demonstrated strong capabilities in solving complex tasks but remain vulnerable when agents receive unreliable messages. This vulnerability stems from a fundamental gap: LLM agents treat all incoming messages equally without evaluating their trustworthiness. While some existing studies approach trustworthiness, they focus on a single type of harmfulness rather than analyze it in a holistic approach from multiple trustworthiness perspectives. We address this gap by proposing a comprehensive definition of trustworthiness inspired by human communication theory (Grice, 1975). Our definition identifies six orthogonal trust dimensions that provide interpretable measures of trustworthiness. Building on this definition, we introduce the Attention Trust Score (A -Trust), a lightweight, attention-based method for evaluating the trustworthiness of messages. We then develop a principled trust management system (TMS) for LLM -MAS that supports both message-level and agent-level trust assessments. Experiments across diverse multi-agent settings and tasks demonstrate that our TMS significantly improves robustness against malicious inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12301v1">Local-Splitter: A Measurement Study of Seven Tactics for Reducing Cloud LLM Token Usage on Coding-Agent Workloads</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      We present a systematic measurement study of seven tactics for reducing cloud LLM token usage when a small local model can act as a triage layer in front of a frontier cloud model. The tactics are: (1) local routing, (2) prompt compression, (3) semantic caching, (4) local drafting with cloud review, (5) minimal-diff edits, (6) structured intent extraction, and (7) batching with vendor prompt caching. We implement all seven in an open-source shim that speaks both MCP and the OpenAI-compatible HTTP surface, supporting any local model via Ollama and any cloud model via an OpenAI-compatible endpoint. We evaluate each tactic individually, in pairs, and in a greedy-additive subset across four coding-agent workload classes (edit-heavy, explanation-heavy, general chat, RAG-heavy). We measure tokens saved, dollar cost, latency, and routing accuracy. Our headline finding is that T1 (local routing) combined with T2 (prompt compression) achieves 45-79% cloud token savings on edit-heavy and explanation-heavy workloads, while on RAG-heavy workloads the full tactic set including T4 (draft-review) achieves 51% savings. We observe that the optimal tactic subset is workload-dependent, which we believe is the most actionable finding for practitioners deploying coding agents today.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16479v3">TRACE: Evaluating Execution Efficiency of LLM-Based Code Translation</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 I wrongly uploaded twice the same paper; see arXiv:2508.11468
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have substantially improved the functional correctness of code translation, the critical dimension of \textit{execution efficiency} remains overlooked. We present \textbf{\textsc{trace}}, the first benchmark to explicitly assess efficiency in LLM-translated code. \textsc{trace} includes 1,000 efficiency-critical tasks across C++, Java, and Python, each augmented with stress tests that reveal efficiency degradations often overlooked by small-scale tests. Using \textsc{trace}, we conduct an extensive evaluation of 28 representative LLMs and highlight several key insights: 1) Correctness is not a reliable proxy for efficiency: the correctness leader \textit{Claude-4-think} achieves only mid-level time efficiency, outperformed by smaller open-source LLMs such as \textit{Qwen2.5-Coder-14B-Instruct}. 2) Inefficiency is both prevalent and patterned: 23.5\% of correct translations exhibit pronounced inefficiency, distributed across algorithmic faults (11.9\%), language construct mismatches (66.4\%), and resource mismanagement (21.7\%). 3) Inference-time prompt strategies bring only modest improvements, suggesting that current LLMs lack intrinsic efficiency awareness. Together, our results establish efficiency as an essential dimension of code translation and position \textsc{trace} as a principled foundation for efficiency-oriented evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10842v2">Resilient Write: A Six-Layer Durable Write Surface for LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      LLM-powered coding agents increasingly rely on tool-use protocols such as the Model Context Protocol (MCP) to read and write files on a developer's workstation. When a write fails - due to content filters, truncation, or an interrupted session - the agent typically receives no structured signal, loses the draft, and wastes tokens retrying blindly. We present Resilient Write, an MCP server that interposes a six-layer durable write surface between the agent and the filesystem. The layers - pre-flight risk scoring, transactional atomic writes, resume-safe chunking, structured typed errors, out-of-band scratchpad storage, and task-continuity handoff envelopes - are orthogonal and independently adoptable. Each layer maps to a concrete failure mode observed during a real agent session in April 2026, in which content-safety filters silently rejected a draft containing redacted API-key prefixes. Three additional tools - chunk preview, format-aware validation, and journal analytics - emerged from using the system to compose this paper. A 186-test suite validates correctness at each layer, and quantitative comparison against naive and defensive baselines shows a 5x reduction in recovery time and a 13x improvement in agent self-correction rate. Resilient Write is open-source under the MIT license.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12285v1">GAM: Hierarchical Graph-based Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 18 pages, 6 figures
    </div>
    <details class="paper-abstract">
      To sustain coherent long-term interactions, Large Language Model (LLM) agents must navigate the tension between acquiring new information and retaining prior knowledge. Current unified stream-based memory systems facilitate context updates but remain vulnerable to interference from transient noise. Conversely, discrete structured memory architectures provide robust knowledge retention but often struggle to adapt to evolving narratives. To address this, we propose GAM, a hierarchical Graph-based Agentic Memory framework that explicitly decouples memory encoding from consolidation to effectively resolve the conflict between rapid context perception and stable knowledge retention. By isolating ongoing dialogue in an event progression graph and integrating it into a topic associative network only upon semantic shifts, our approach minimizes interference while preserving long-term consistency. Additionally, we introduce a graph-guided, multi-factor retrieval strategy to enhance context precision. Experiments on LoCoMo and LongDialQA indicate that our method consistently outperforms state-of-the-art baselines in both reasoning accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02553v2">Exploring Individual Factors in the Adoption of LLMs for Specific Software Engineering Purposes</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Context: The advent of Large Language Models (LLMs) is transforming software development, significantly enhancing software engineering (SE) processes. Research has explored their role within development teams, focusing on the specific purposes for which LLMs are used within SE tasks, such as artifact generation, decision-making support, and information retrieval. Despite the growing body of work on LLMs in SE, most studies have centered on broad adoption trends, neglecting the nuanced relationship between individual cognitive and behavioral factors and their impact on purpose-specific adoption. While factors such as perceived effort and performance expectancy have been explored at a general level, their influence on distinct SE purposes remains underexamined. This gap hinders the development of tailored LLM-based systems (e.g., Generative AI Agents) that align with engineers' specific needs and limits the ability of team leaders to devise effective strategies for fostering LLM adoption in targeted workflows. Objectives: For the reasons mentioned above, this study aims to study the individual factors that drive the choice to use LLMs for distinct SE purposes. Methods: To achieve the above-mentioned objective, we surveyed 188 software engineers to test the relationship between individual attributes related to technology adoption and LLM adoption across five key purposes, using structural equation modeling (SEM). The Unified Theory of Acceptance and Use of Technology (UTAUT2) was applied to characterize individual adoption behaviors. Results: The findings reveal that purpose-specific adoption is influenced by distinct factors, some of which negatively impact adoption when considered in isolation, underscoring the complexity of LLM integration in SE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12268v1">CodeSpecBench: Benchmarking LLMs for Executable Behavioral Specification Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate code from natural language, but the extent to which they capture intended program behavior remains unclear. Executable behavioral specifications, defined via preconditions and postconditions, provide a concrete means to assess such understanding. However, existing work on specification generation is constrained in evaluation methodology, task settings, and specification expressiveness. We introduce CodeSpecBench, a benchmark for executable behavioral specification generation under an execution-based evaluation protocol. CodeSpecBench supports both function-level and repository-level tasks and encodes specifications as executable Python functions. Constructed from diverse real-world codebases, it enables a realistic assessment of both correctness (accepting valid behaviors) and completeness (rejecting invalid behaviors). Evaluating 15 state-of-the-art LLMs on CodeSpecBench, we observe a sharp performance degradation on repository-level tasks, where the best model attains only a 20.2% pass rate. We further find that specification generation is substantially more challenging than code generation, indicating that strong coding performance does not necessarily reflect deep understanding of intended program semantics. Our data and code are available at https://github.com/SparksofAGI/CodeSpecBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12262v1">CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 12 pages, 6 figures, 4 tables, 1 algorithm
    </div>
    <details class="paper-abstract">
      Cascaded LLM systems coordinate models of varying sizes with human experts to balance accuracy, cost, and abstention under uncertainty. However, single-model tiers at each stage often struggle with ambiguous queries, triggering premature escalations to costlier models or experts due to under-confidence and inefficient compute scaling. CascadeDebate addresses this gap by inserting multi-agent deliberation directly at each tier's escalation boundary. Confidence-based routers activate lightweight agent ensembles only for uncertain cases, enabling consensus-driven resolution of ambiguities internally without invoking higher-cost upgrades. Our unified architecture alternates single-model inference with selective multi-agent deliberation across model scales, culminating in human experts as the final fallback. This design scales test-time compute dynamically according to query difficulty. Across five benchmarks spanning science, medicine, and general knowledge, CascadeDebate outperforms strong single-model cascades and standalone multi-agent systems by up to 26.75 percent. An online threshold optimizer proves essential, boosting accuracy by 20.98 to 52.33 percent relative improvement over fixed policies and enabling elastic adaptation to real-world distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12250v1">How memory can affect collective and cooperative behaviors in an LLM-Based Social Particle Swarm</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 12 pages, 6 figures and 2 tables
    </div>
    <details class="paper-abstract">
      This study examines how model-specific characteristics of Large Language Model (LLM) agents, including internal alignment, shape the effect of memory on their collective and cooperative dynamics in a multi-agent system. To this end, we extend the Social Particle Swarm (SPS) model, in which agents move in a two-dimensional space and play the Prisoner's Dilemma with neighboring agents, by replacing its rule-based agents with LLM agents endowed with Big Five personality scores and varying memory lengths. Using Gemini-2.0-Flash, we find that memory length is a critical parameter governing collective behavior: even a minimal memory drastically suppressed cooperation, transitioning the system from stable cooperative clusters through cyclical formation and collapse of clusters to a state of scattered defection as memory length increased. Big Five personality traits correlated with agent behaviors in partial agreement with findings from experiments with human participants, supporting the validity of the model. Comparative experiments using Gemma~3:4b revealed the opposite trend: longer memory promoted cooperation, accompanied by the formation of dense cooperative clusters. Sentiment analysis of agents' reasoning texts showed that Gemini interprets memory increasingly negatively as its length grows, while Gemma interprets it less negatively, and that this difference persists in the early phase of experiments before the macro-level dynamics converge. These results suggest that model-specific characteristics of LLMs, potentially including alignment, play a fundamental role in determining emergent social behavior in Generative Agent-Based Modeling, and provide a micro-level cognitive account of the contradictions found in prior work on memory and cooperation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23200v2">From Coordinates to Context: An LLM-Bootstrapped Semantic Encoding Framework for Privacy-Preserving Mobile Sensing Stress Recognition</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Psychological stress is a widespread issue that significantly impacts student well-being and academic performance. Effective remote stress recognition is crucial, yet existing methods often rely on wearable devices or GPS-based clustering techniques that pose privacy risks and lack of human understandable explanations. In this study, we introduce a novel, end-to-end privacy-enhanced framework for semantic location encoding using a self-hosted OSM engine and an LLM-bootstrapped static map for human-friendly feature extraction, and pave a pathway for privacy-aware location data transformation for dataset sharing. We rigorously quantify the privacy-utility-explainability trilemma and demonstrate (via LOSO validation) that our Privacy-Aware (PA) model achieves robust privacy protection without being statistically distinguishable in stress recognition performance from a non-private model. Model explanation analysis highlights that our extracted features, which are user-friendly features, match with psychological literature about stress. In addition, an ablation study on the GeoLife dataset also demonstrates that our privacy framework improves privacy by 2-3 times compared to a non-privacy-aware approach. This suggests that our system can be utilized for the next generation of GPS transformations in open-source datasets for future researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07248v2">DarwinTOD: LLM-driven Lifelong Self-evolution for Task-oriented Dialog Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted in ACL2026 main
    </div>
    <details class="paper-abstract">
      Traditional task-oriented dialog systems are unable to evolve from ongoing interactions or adapt to new domains after deployment, that is a critical limitation in real-world dynamic environments. Continual learning approaches depend on episodic retraining with human curated data, failing to achieve autonomy lifelong improvement. While evolutionary computation and LLM driven self improvement offer promising mechanisms for dialog optimization, they lack a unified framework for holistic, iterative strategy refinement. To bridge this gap, we propose DarwinTOD, a lifelong self evolving dialog framework that systematically integrates these two paradigms, enabling continuous strategy optimization from a zero-shot base without task specific fine-tuning. DarwinTOD maintains an Evolvable Strategy Bank and operates through a dual-loop process: online multi-agent dialog execution with peer critique, and offline structured evolutionary operations that refine the strategy bank using accumulated feedback. This closed-loop design enables autonomous continuous improvement without human intervention. Extensive experiments show that DarwinTOD surpasses previous state-of-the-art methods and exhibits continuous performance gains throughout evolution. Our work provides a novel framework for building dialog systems with lifelong self evolution capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12232v1">TEMPLATEFUZZ: Fine-Grained Chat Template Fuzzing for Jailbreaking and Red Teaming LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across diverse domains, yet their vulnerability to jailbreak attacks, where adversarial inputs bypass safety mechanisms to elicit harmful outputs, poses significant security risks. While prior work has primarily focused on prompt injection attacks, these approaches often require resource-intensive prompt engineering and overlook other critical components, such as chat templates. This paper introduces TEMPLATEFUZZ, a fine-grained fuzzing framework that systematically exposes vulnerabilities in chat templates, a critical yet underexplored attack surface in LLMs. Specifically, TEMPLATEFUZZ (1) designs a series of element-level mutation rules to generate diverse chat template variants, (2) proposes a heuristic search strategy to guide the chat template generation toward the direction of amplifying the attack success rate (ASR) while preserving model accuracy, and (3) integrates an active learning-based strategy to derive a lightweight rule-based oracle for accurate and efficient jailbreak evaluation. Evaluated on twelve open-source LLMs across multiple attack scenarios, TEMPLATEFUZZ achieves an average ASR of 98.2% with only 1.1% accuracy degradation, outperforming state-of-the-art methods by 9.1%-47.9% in ASR and 8.4% in accuracy degradation. Moreover, even on five industry-leading commercial LLMs where chat templates cannot be specified, TEMPLATEFUZZ attains a 90% average ASR via chat template-based prompt injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12228v1">From IOCs to Regex: Automating CTI Operationalization for SOC with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Cyber Threat Intelligence (CTI) reports contain Indicators of Compromise (IOCs) that are critical for security operations. To operationalize these IOCs across heterogeneous logs, analysts often convert them into regular expressions (regexes) for tasks such as digital forensics, log parsing, and SIEM rule creation. However, regex construction is still largely manual, requiring analysts to extract IOCs from CTI reports and transform them into syntactically valid and semantically precise patterns. This process is slow, error-prone, and increasingly impractical as CTI volumes grow. Although recent studies have applied Large Language Models (LLMs) to IOC extraction, they typically output plain strings rather than regexes, limiting practical deployment. Plain IOCs cannot effectively capture variations in system context, log format, or attacker behavior. To address this gap, we propose IOCRegex-gen, a fully automated LLM-based regex generation system that converts IOCs into regexes. The system introduces two key innovations: (i) a group-aware mechanism that identifies which IOC segments should be represented as capture or non-capture groups, and (ii) an iterative reasoning and multi-stage validation pipeline to ensure syntactic validity and semantic correctness. Experiments on over 3,000 real CTI reports and 2,400 ground-truth strings from the MITRE ATT&CK Evaluation framework show that IOCRegex-gen achieves an average hit rate of 99.1% and a false-positive rate of only 0.8%, demonstrating its effectiveness for large-scale CTI processing and automated regex generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12227v1">Designing Reliable LLM-Assisted Rubric Scoring for Constructed Responses: Evidence from Physics Exams</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Student responses in STEM assessments are often handwritten and combine symbolic expressions, calculations, and diagrams, creating substantial variation in format and interpretation. Despite their importance for evaluating students' reasoning, such responses are time-consuming to score and prone to rater inconsistency, particularly when partial credit is required. Recent advances in large language models (LLMs) have increased attention to AI-assisted scoring, yet evidence remains limited regarding how rubric design and LLM configurations influence reliability across performance levels. This study examined the reliability of AI-assisted scoring of undergraduate physics constructed responses using GPT-4o. Twenty authentic handwritten exam responses were scored across two rounds by four instructors and by the AI model using skill-based rubrics with differing levels of analytic granularity. Prompting format and temperature settings were systematically varied. Overall, human-AI agreement on total scores was comparable to human inter-rater reliability and was highest for high- and low-performing responses, but declined for mid-level responses involving partial or ambiguous reasoning. Criterion-level analyses showed stronger alignment for clearly defined conceptual skills than for extended procedural judgments. A more fine-grained, checklist-based rubric improved consistency relative to holistic scoring. These findings indicate that reliable AI-assisted scoring depends primarily on clear, well-structured rubrics, while prompting format plays a secondary role and temperature has relatively limited impact. More broadly, the study provides transferable design recommendations for implementing reliable LLM-assisted scoring in STEM contexts through skill-based rubrics and controlled LLM settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12223v1">LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to Findings of the Association for Computational Linguistics (ACL 2026)
    </div>
    <details class="paper-abstract">
      Pretrained language models (PLMs) like BERT provide strong semantic representations but are costly and opaque, while symbolic models such as the Tsetlin Machine (TM) offer transparency but lack semantic generalization. We propose a semantic bootstrapping framework that transfers LLM knowledge into symbolic form, combining interpretability with semantic capacity. Given a class label, an LLM generates sub-intents that guide synthetic data creation through a three-stage curriculum (seed, core, enriched), expanding semantic diversity. A Non-Negated TM (NTM) learns from these examples to extract high-confidence literals as interpretable semantic cues. Injecting these cues into real data enables a TM to align clause logic with LLM-inferred semantics. Our method requires no embeddings or runtime LLM calls, yet equips symbolic models with pretrained semantic priors. Across multiple text classification tasks, it improves interpretability and accuracy over vanilla TM, achieving performance comparable to BERT while remaining fully symbolic and efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12218v1">LLM-Enhanced Log Anomaly Detection: A Comprehensive Benchmark of Large Language Models for Automated System Diagnostics</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 5 pages, 4 tables, code available at https://github.com/disha8611/llm-log-anomaly-benchmark
    </div>
    <details class="paper-abstract">
      System log anomaly detection is critical for maintaining the reliability of large-scale software systems, yet traditional methods struggle with the heterogeneous and evolving nature of modern log data. Recent advances in Large Language Models (LLMs) offer promising new approaches to log understanding, but a systematic comparison of LLM-based methods against established techniques remains lacking. In this paper, we present a comprehensive benchmark study evaluating both LLM-based and traditional approaches for log anomaly detection across four widely-used public datasets: HDFS, BGL, Thunderbird, and Spirit. We evaluate three categories of methods: (1) classical log parsers (Drain, Spell, AEL) combined with machine learning classifiers, (2) fine-tuned transformer models (BERT, RoBERTa), and (3) prompt-based LLM approaches (GPT-3.5, GPT-4, LLaMA-3) in zero-shot and few-shot settings. Our experiments reveal that while fine-tuned transformers achieve the highest F1-scores (0.96-0.99), prompt-based LLMs demonstrate remarkablezero-shot capabilities (F1: 0.82-0.91) without requiring any labeled training data -- a significant advantage for real-world deployment where labeled anomalies are scarce. We further analyze the cost-accuracy trade-offs, latency characteristics, and failure modes of each approach. Our findings provide actionable guidelines for practitioners choosing log anomaly detection methods based on their specific constraints regarding accuracy, latency, cost, and label availability. All code and experimental configurations are publicly available to facilitate reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11120v2">Persona Non Grata: Single-Method Safety Evaluation Is Incomplete for Persona-Imbued LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Personality imbuing customizes LLM behavior, but safety evaluations almost always study prompt-based personas alone. We show this is incomplete: prompting and activation steering expose *different*, architecture-dependent vulnerability profiles, and testing with only one method can miss a model's dominant failure mode. Across 5,568 judged conditions on four standard models from three architecture families, persona danger rankings under system prompting are preserved across all architectures ($ρ= 0.71$--$0.96$), but activation-steering vulnerability diverges sharply and cannot be predicted from prompt-side rankings: Llama-3.1-8B is substantially more AS-vulnerable, whereas Gemma-3-27B and Qwen3.5 are more vulnerable to prompting. The most striking illustration of this divergence is the *prosocial persona paradox*: on Llama-3.1-8B, P12 (high conscientiousness + high agreeableness) is among the safest personas under prompting yet becomes the highest-ASR activation-steered persona (ASR ~0.818). This is an inversion robust to coefficient ablation and matched-strength calibration, and replicated on DeepSeek-R1-Distill-Qwen-32B. A trait refusal alignment framework, in which conscientiousness is strongly anti-aligned with refusal on Llama-3.1-8B, offers a partial geometric account. Reasoning provides only partial protection: two 32B reasoning models reach 15--18% prompt-side ASR, and activation steering separates them sharply in both baseline susceptibility and persona-specific vulnerability. Heuristic trace diagnostics suggest that the safer model retains stronger policy recall and self-correction behavior, not merely longer reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12201v1">AdversarialCoT: Single-Document Retrieval Poisoning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 6 pages,accepted by SIGIR 2026 short paper
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language model (LLM) reasoning by retrieving external documents, but also opens up new attack surfaces. We study knowledge-base poisoning attacks in RAG, where an attacker injects malicious content into the retrieval corpus, which is then naturally surfaced by the retriever and consumed by the LLM during reasoning. Unlike prior work that floods the corpus with poisoned documents, we propose AdversarialCoT, a query-specific attack that poisons only a single document in the corpus. AdversarialCoT first extracts the target LLM's reasoning framework to guide the construction of an initial adversarial chain-of-thought (CoT). The adversarial document is iteratively refined through interactions with the LLM, progressively exposing and exploiting critical reasoning vulnerabilities. Experiments on benchmark LLMs show that a single adversarial document can significantly degrade reasoning accuracy, revealing subtle yet impactful weaknesses. This study exposes security risks in RAG systems and provides actionable insights for designing more robust LLM reasoning pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13132v1">LLM-Driven Large-Scale Spectrum Access</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 11 pages, 2 figures, 8 tables. Submitted to IEEE Transactions on Mobile Computing (TMC)
    </div>
    <details class="paper-abstract">
      Efficient spectrum management in massive-scale wireless networks is increasingly challenged by explosive action spaces and the computational intractability of traditional optimization. This study proposes a Large-Scale LLM-Driven Spectrum Access (LSA) framework rooted in Group Relative Policy Optimization (GRPO). To overcome the computational collapse caused by ultra-long prompts in large-scale scenarios, we develop a hierarchical state serialization mechanism that synthesizes global environment statistics with localized critical constraints, enabling the LLM to perform high-dimensional reasoning within a bounded context window. Simulation results under strictly time-bounded inference protocols reveal that the code-driven paradigm eliminates the SFT cold-start bottleneck and leverages direct execution feedback to achieve superior scaling laws. The framework maintains robust spectral utility and generalization across varying network scales, yielding consistent and empirically superior performance over non-deterministic heuristics, and surpassing partitioned classical solvers in ultra-dense regimes under matched compute budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12198v1">Towards grounded autonomous research: an end-to-end LLM mini research loop on published computational physics</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Recent autonomous LLM agents have demonstrated end-to-end automation of machine-learning research. Real-world physical science is intrinsically harder, requiring deep reasoning bounded by physical truth and, because real systems are too complex to study in isolation, almost always built on existing literature. We focus on the smallest meaningful unit of such research, a mini research loop in which an agent reads a paper, reproduces it, critiques it, and extends it. We test this loop in two complementary regimes: scale and depth. At scale, across 111 open-access computational physics papers, an agent autonomously runs the read-plan-compute-compare loop and, without being asked to critique, raises substantive concerns on ~42% of papers - 97.7% of which require execution to surface. In depth, for one Nature Communications paper on multiscale simulation of a 2D-material MOSFET, the agent runs new calculations missing from the original and produces, unsupervised, a publishable Comment -- composed, figured, typeset, and PDF-iterated -- that revises the paper's headline conclusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00136v2">ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 27 pages, 15 figures, 13 tables. Code available at https://github.com/ParetoBandit/ParetoBandit
    </div>
    <details class="paper-abstract">
      Multi-model LLM serving operates in a non-stationary, noisy environment: providers revise pricing, model quality can shift or regress without notice, and new models arrive regularly. More than a dozen recent methods have proposed learned routers to navigate the resulting quality--cost tradeoff across portfolios spanning a $\sim$530$\times$ cost range. Despite this activity, two gaps in the current solution space limit routing effectiveness under these conditions: no existing router enforces a dollar-denominated cost ceiling in closed loop over an open-ended request stream, and none provides principled online adaptation to post-deployment shifts in pricing or model quality. We present ParetoBandit, an open-source adaptive router built on cost-aware contextual bandits that addresses both gaps. Its core contributions are: (1) an online primal--dual budget pacer that enforces a per-request cost ceiling without a known horizon, and (2) geometric forgetting on sufficient statistics that gives the bandit bounded memory for tracking quality and cost shifts. A hot-swap model registry further supports runtime model changes with budget-controlled exploration. On 1,824 benchmark prompts with a three-model portfolio, the router maintains budget compliance within 0.4%, adapts to price and quality shifts with up to +0.071 quality lift, and integrates a cold-started model within $\sim$142 steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19693v2">Beyong Tokens: Item-aware Attention for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently gained increasing attention in the field of recommendation. Existing LLM-based methods typically represent items as token sequences, and apply attention layers on these tokens to generate recommendations. However, by inheriting the standard attention mechanism, these methods focus on modeling token-level relations. This token-centric focus overlooks the item as the fundamental unit of recommendation, preventing existing methods from effectively capturing collaborative relations at the item level. In this work, we revisit the role of tokens in LLM-driven recommendation and categorize their relations into two types: (1) intra-item token relations, which present the content semantics of an item, e.g., name, color, and size; and (2) inter-item token relations, which encode collaborative relations across items. Building on these insights, we propose a novel framework with an item-aware attention mechanism (IAM) to enhance LLMs for recommendation. Specifically, IAM devises two complementary attention layers: (1) an intra-item attention layer, which restricts attention to tokens within the same item, modeling item content semantics; and (2) an inter-item attention layer, which attends exclusively to token relations across items, capturing item collaborative relations. Through this stacked design, IAM explicitly emphasizes items as the fundamental units in recommendation, enabling LLMs to effectively exploit item-level collaborative relations. Extensive experiments on several public datasets demonstrate the effectiveness of IAM in enhancing LLMs for personalized recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12179v1">AgenticAI-DialogGen: Topic-Guided Conversation Generation for Fine-Tuning and Evaluating Short- and Long-Term Memories of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 13 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have improved their ability to process extended conversational contexts, yet fine-tuning and evaluating short- and long-term memories remain difficult due to the absence of datasets that encode both short- and long-term conversational history. Existing conversational datasets lack memory grounding, overlook topic continuity, or rely on costly human annotation. To address these gaps, we introduce AgenticAI-DialogGen, a modular agent-based framework that generates persona-grounded and topic-guided conversations without human supervision. The framework uses LLM agents to extract knowledge graphs, identify topics, build speaker personas, and simulate topic-guided conversations from unstructured conversations. A QA module generates memory-grounded Question Answer (QA) pairs drawn from short- and long-term conversational histories. We also generated a new dataset entitled, TopicGuidedChat (TGC), where long-term memory is encoded as speaker-specific knowledge graphs and short-term memory as newly generated topic-guided conversations. Evaluations depict that AgenticAI-DialogGen yields higher conversational quality and LLMs fine-tuned on TGC dataset achieve improved performance on memory-grounded QA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12177v1">Policy-Invisible Violations in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 26 pages,1 figure, 11 tables
    </div>
    <details class="paper-abstract">
      LLM-based agents can execute actions that are syntactically valid, user-sanctioned, and semantically appropriate, yet still violate organizational policy because the facts needed for correct policy judgment are hidden at decision time. We call this failure mode policy-invisible violations: cases in which compliance depends on entity attributes, contextual state, or session history absent from the agent's visible context. We present PhantomPolicy, a benchmark spanning eight violation categories with balanced violation and safe-control cases, in which all tool responses contain clean business data without policy metadata. We manually review all 600 model traces produced by five frontier models and evaluate them using human-reviewed trace labels. Manual review changes 32 labels (5.3%) relative to the original case-level annotations, confirming the need for trace-level human review. To demonstrate what world-state-grounded enforcement can achieve under favorable conditions, we introduce Sentinel, an enforcement framework based on counterfactual graph simulation. Sentinel treats every agent action as a proposed mutation to an organizational knowledge graph, performs speculative execution to materialize the post-action world state, and verifies graph-structural invariants to decide Allow/Block/Clarify. Against human-reviewed trace labels, Sentinel substantially outperforms a content-only DLP baseline (68.8% vs. 93.0% accuracy) while maintaining high precision, though it still leaves room for improvement on certain violation categories. These results demonstrate what becomes achievable once policy-relevant world state is made available to the enforcement layer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12176v1">Evaluating Relational Reasoning in LLMs with REL</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 9 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Relational reasoning is the ability to infer relations that jointly bind multiple entities, attributes, or variables. This ability is central to scientific reasoning, but existing evaluations of relational reasoning in large language models often focus on structured inputs such as tables, graphs, or synthetic tasks, and do not isolate the difficulty introduced by higher-arity relational binding. We study this problem through the lens of Relational Complexity (RC), which we define as the minimum number of independent entities or operands that must be simultaneously bound to apply a relation. RC provides a principled way to vary reasoning difficulty while controlling for confounders such as input size, vocabulary, and representational choices. Building on RC, we introduce REL, a generative benchmark framework spanning algebra, chemistry, and biology that varies RC within each domain. Across frontier LLMs, performance degrades consistently and monotonically as RC increases, even when the total number of entities is held fixed. This failure mode persists with increased test-time compute and in-context learning, suggesting a limitation tied to the arity of the required relational binding rather than to insufficient inference steps or lack of exposure to examples. Our results identify a regime of higher-arity reasoning in which current models struggle, and motivate re-examining benchmarks through the lens of relational complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.16057v5">Neuro-symbolic Static Analysis with LLM-generated Vulnerability Patterns</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      In this work, we present MoCQ, a neuro-symbolic static analysis framework that leverages large language models (LLMs) to automatically generate vulnerability detection patterns. This approach combines the precision and scalability of pattern-based static analysis with the semantic understanding and automation capabilities of LLMs. MoCQ extracts the domain-specific languages for expressing vulnerability patterns and employs an iterative refinement loop with trace-driven symbolic validation that provides precise feedback for pattern correction. We evaluated MoCQ on 12 vulnerability types across four languages (C/C++, Java, PHP, JavaScript). MoCQ achieves detection performance comparable to expert-developed patterns while requiring only hours of generation versus weeks of manual effort. Notably, MoCQ uncovered 46 new vulnerability patterns that security experts had missed and discovered 25 previously unknown vulnerabilities in real-world applications. MoCQ also outperforms prior approaches with stronger analysis capabilities and broader applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08491v1">Figures as Interfaces: Toward LLM-Native Artifacts for Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming scientific workflows, not only through their generative capabilities but also through their emerging ability to use tools, reason about data, and coordinate complex analytical tasks. Yet in most human-AI collaborations, the primary outputs, figures, are still treated as static visual summaries: once rendered, they are handled by both humans and multimodal LLMs as images to be re-interpreted from pixels or captions. The emergent capabilities of LLMs open an opportunity to fundamentally rethink this paradigm. In this paper, we introduce the concept of LLM-native figures: data-driven artifacts that are simultaneously human-legible and machine-addressable. Unlike traditional plots, each artifact embeds complete provenance: the data subset, analytical operations and code, and visualization specification used to generate it. As a result, an LLM can "see through" the figure--tracing selections back to their sources, generating code to extend analyses, and orchestrating new visualizations through natural-language instructions or direct manipulation. We implement this concept through a hybrid language-visual interface that integrates LLM agents with a bidirectional mapping between figures and underlying data. Using the science of science domain as a testbed, we demonstrate that LLM-native figures can accelerate discovery, improve reproducibility, and make reasoning transparent across agents and users. More broadly, this work establishes a general framework for embedding provenance, interactivity, and explainability into the artifacts of modern research, redefining the figure not as an end product, but as an interface for discovery. For more details, please refer to the demo video available at www.llm-native-figure.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08477v1">SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 23 Pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at https://github.com/asuvarna31/supernova.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08417v1">Vulnerability Detection with Interprocedural Context in Multiple Languages: Assessing Effectiveness and Cost of Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been a promising way for automated vulnerability detection. However, most prior studies have explored the use of LLMs to detect vulnerabilities only within single functions, disregarding those related to interprocedural dependencies. These studies overlook vulnerabilities that arise from data and control flows that span multiple functions. Thus, leveraging the context provided by callers and callees may help identify vulnerabilities. This study empirically investigates the effectiveness of detection, the inference cost, and the quality of explanations of four modern LLMs (Claude Haiku 4.5, GPT-4.1 Mini, GPT-5 Mini, and Gemini 3 Flash) in detecting vulnerabilities related to interprocedural dependencies. To do that, we conducted an empirical study on 509 vulnerabilities from the ReposVul dataset, systematically varying the level of interprocedural context (target function code-only, target function + callers, and target function + callees) and evaluating the four modern LLMs across C, C++, and Python. The results show that Gemini 3 Flash offers the best cost-effectiveness trade-off for C vulnerabilities, achieving F1 >= 0.978 at an estimated cost of $0.50-$0.58 per configuration, and Claude Haiku 4.5 correctly identified and explained the vulnerability in 93.6% of the evaluated cases. Overall, the findings have direct implications for the design of AI-assisted security analysis tools that can generalize across codebases in multiple programming languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08407v1">Your Agent Is Mine: Measuring Malicious Intermediary Attacks on the LLM Supply Chain</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on third-party API routers to dispatch tool-calling requests across multiple upstream providers. These routers operate as application-layer proxies with full plaintext access to every in-flight JSON payload, yet no provider enforces cryptographic integrity between client and upstream model. We present the first systematic study of this attack surface. We formalize a threat model for malicious LLM API routers and define two core attack classes, payload injection (AC-1) and secret exfiltration (AC-2), together with two adaptive evasion variants: dependency-targeted injection (AC-1.a) and conditional delivery (AC-1.b). Across 28 paid routers purchased from Taobao, Xianyu, and Shopify-hosted storefronts and 400 free routers collected from public communities, we find 1 paid and 8 free routers actively injecting malicious code, 2 deploying adaptive evasion triggers, 17 touching researcher-owned AWS canary credentials, and 1 draining ETH from a researcher-owned private key. Two poisoning studies further show that ostensibly benign routers can be pulled into the same attack surface: a leaked OpenAI key generates 100M GPT-5.4 tokens and more than seven Codex sessions, while weakly configured decoys yield 2B billed tokens, 99 credentials across 440 Codex sessions, and 401 sessions already running in autonomous YOLO mode. We build Mine, a research proxy that implements all four attack classes against four public agent frameworks, and use it to evaluate three deployable client-side defenses: a fail-closed policy gate, response-side anomaly screening, and append-only transparency logging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08401v1">Verify Before You Commit: Towards Faithful Reasoning in LLM Agents via Self-Auditing</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted by ACL2026 Main Conference
    </div>
    <details class="paper-abstract">
      In large language model (LLM) agents, reasoning trajectories are treated as reliable internal beliefs for guiding actions and updating memory. However, coherent reasoning can still violate logical or evidential constraints, allowing unsupported beliefs repeatedly stored and propagated across decision steps, leading to systematic behavioral drift in long-horizon agentic systems. Most existing strategies rely on the consensus mechanism, conflating agreement with faithfulness. In this paper, inspired by the vulnerability of unfaithful intermediate reasoning trajectories, we propose \textbf{S}elf-\textbf{A}udited \textbf{Ve}rified \textbf{R}easoning (\textsc{SAVeR}), a novel framework that enforces verification over internal belief states within the agent before action commitment, achieving faithful reasoning. Concretely, we structurally generate persona-based diverse candidate beliefs for selection under a faithfulness-relevant structure space. To achieve reasoning faithfulness, we perform adversarial auditing to localize violations and repair through constraint-guided minimal interventions under verifiable acceptance criteria. Extensive experiments on six benchmark datasets demonstrate that our approach consistently improves reasoning faithfulness while preserving competitive end-task performance.
    </details>
</div>
