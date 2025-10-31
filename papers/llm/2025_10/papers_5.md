# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19507v1">Teaming LLMs to Detect and Mitigate Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted to NeurIPS 2025 workshop on Reliable ML from Unreliable Data
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated state-of-the-art results in large language model (LLM) hallucination detection and mitigation through consistency-based approaches which involve aggregating multiple responses sampled from a single LLM for a given prompt. These approaches help offset limitations stemming from the imperfect data on which LLMs are trained, which includes biases and under-representation of information required at deployment time among other limitations which can lead to hallucinations. We show that extending these single-model consistency methods to combine responses from multiple LLMs with different training data, training schemes and model architectures can result in substantial further improvements in hallucination detection and mitigation capabilities beyond their single-model consistency counterparts. We evaluate this \emph{consortium consistency} approach across many model teams from a pool of 15 LLMs and explore under what conditions it is beneficial to team together different LLMs in this manner. Further, we show that these performance improvements often come with reduced inference costs, offsetting a significant drawback with single-model consistency methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v5">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted by EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19462v1">AegisMCP: Online Graph Intrusion Detection for Tool-Augmented LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      In this work, we study security of Model Context Protocol (MCP) agent toolchains and their applications in smart homes. We introduce AegisMCP, a protocol-level intrusion detector. Our contributions are: (i) a minimal attack suite spanning instruction-driven escalation, chain-of-tool exfiltration, malicious MCP server registration, and persistence; (ii) NEBULA-Schema (Network-Edge Behavioral Learning for Untrusted LLM Agents), a reusable protocol-level instrumentation that represents MCP activity as a streaming heterogeneous temporal graph over agents, MCP servers, tools, devices, remotes, and sessions; and (iii) a CPU-only streaming detector that fuses novelty, session-DAG structure, and attribute cues for near-real-time edge inference, with optional fusion of local prompt-guardrail signals. On an emulated smart-home testbed spanning multiple MCP stacks and a physical bench, AegisMCP achieves sub-second per-window model inference and end-to-end alerting. The latency of AegisMCP is consistently sub-second on Intel N150-class edge hardware, while outperforming traffic-only and sequence baselines; ablations confirm the importance of DAG and install/permission signals. We release code, schemas, and generators for reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19438v1">AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Autonomous Driving Systems (ADS) are safety-critical, where failures can be severe. While Metamorphic Testing (MT) is effective for fault detection in ADS, existing methods rely heavily on manual effort and lack automation. We present AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that automates the extraction of Metamorphic Relations (MRs) from local traffic rules and the generation of valid follow-up test cases. AutoMT leverages LLMs to extract MRs from traffic rules in Gherkin syntax using a predefined ontology. A vision-language agent analyzes scenarios, and a search agent retrieves suitable MRs from a RAG-based database to generate follow-up cases via computer vision. Experiments show that AutoMT achieves up to 5 x higher test diversity in follow-up case generation compared to the best baseline (manual expert-defined MRs) in terms of validation rate, and detects up to 20.55% more behavioral violations. While manual MT relies on a fixed set of predefined rules, AutoMT automatically extracts diverse metamorphic relations that augment real-world datasets and help uncover corner cases often missed during in-field testing and data collection. Its modular architecture separating MR extraction, filtering, and test generation supports integration into industrial pipelines and potentially enables simulation-based testing to systematically cover underrepresented or safety-critical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19420v1">Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16062v2">Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 47 pages, 25 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: https://correctbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01349v4">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19361v1">AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21625v6">Meeseeks: A Feedback-Driven, Iterative Self-Correction Benchmark evaluating LLMs' Instruction Following Capability</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The capability to precisely adhere to instructions is a cornerstone for Large Language Models (LLMs) to function as dependable agents in real-world scenarios. However, confronted with complex prompts, LLMs frequently encounter difficulties in fulfilling all specified requirements within a single response. Drawing inspiration from recent advancements in Chain-of-Thought (CoT) prompting and self-correction methodologies, we introduce Meeseeks (The name is inspired by Mr. Meeseeks from "Rick and Morty," a character renowned for efficiently accomplishing assigned tasks. See: https://en.wikipedia.org/wiki/Mr._Meeseeks), a fully automated iterative instruction-following benchmark equipped with an integrated feedback mechanism. Meeseeks identifies erroneous components in model responses and provides corresponding feedback accurately, thereby iteratively guiding the model toward self-correction. The dataset contains over 700 curated instances annotated by 32 distinct capability tags in Chinese and English. Extensive experimental results reveal that different state-of-the-art commercial and open-source LLMs exhibit vastly disparate performance, and even after 20 turns of iterative feedback-driven self-correction, nearly all models demonstrate suboptimal performance. We conducted comprehensive analysis from both macro and instance levels, uncovering numerous common issues prevalent in current state-of-the-art models, as well as several counterintuitive phenomena. We've open-sourced our work on https://github.com/ADoublLEN/Meeseeks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19331v1">Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 This paper has been accepted for the upcoming 59th Hawaii International Conference on System Sciences (HICSS-59), 2026, Hawaii, USA. The final published version will appear in the official conference proceedings
    </div>
    <details class="paper-abstract">
      In this paper, we investigate how personalising Large Language Models (Persona-LLMs) with annotator personas affects their sensitivity to hate speech, particularly regarding biases linked to shared or differing identities between annotators and targets. To this end, we employ Google's Gemini and OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona prompting and a deeply contextualised persona development based on Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We analyse the impact of using in-group and out-group annotator personas on the models' detection performance and fairness across diverse social groups. This work bridges psychological insights on group identity with advanced NLP techniques, demonstrating that incorporating socio-demographic attributes into LLMs can address bias in automated hate speech detection. Our results highlight both the potential and limitations of persona-based approaches in reducing bias, offering valuable insights for developing more equitable hate speech detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14000v2">Type-aware LLM-based Regression Test Generation for Python Programs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Automated regression test generation has been extensively explored, yet generating high-quality tests for Python programs remains particularly challenging. Because of the Python's dynamic typing features, existing approaches, ranging from search-based software testing (SBST) to recent LLM-driven techniques, are often prone to type errors. Hence, existing methods often generate invalid inputs and semantically inconsistent test cases, which ultimately undermine their practical effectiveness. To address these limitations, we present Test4Py, a novel framework that enhances type correctness in automated test generation for Python. Test4Py leverages the program's call graph to capture richer contextual information about parameters, and introduces a behavior-based type inference mechanism that accurately infers parameter types and construct valid test inputs. Beyond input construction, Test4Py integrates an iterative repair procedure that progressively refines generated test cases to improve coverage. In an evaluation on 183 real-world Python modules, Test4Py achieved an average statement coverage of 83.0% and branch coverage of 70.8%, outperforming state-of-the-art tools by 7.2% and 8.4%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19327v1">SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The rapid evolution of smart cities has increased the reliance on intelligent interconnected services to optimize infrastructure, resources, and citizen well-being. Agentic AI has emerged as a key enabler by supporting autonomous decision-making and adaptive coordination, allowing urban systems to respond in real time to dynamic conditions. Its benefits are evident in areas such as transportation, where the integration of traffic data, weather forecasts, and safety sensors enables dynamic rerouting and a faster response to hazards. However, its deployment across heterogeneous smart city ecosystems raises critical governance, risk, and compliance (GRC) challenges, including accountability, data privacy, and regulatory alignment within decentralized infrastructures. Evaluation of SORA-ATMAS with three domain agents (Weather, Traffic, and Safety) demonstrated that its governance policies, including a fallback mechanism for high-risk scenarios, effectively steer multiple LLMs (GPT, Grok, DeepSeek) towards domain-optimized, policy-aligned outputs, producing an average MAE reduction of 35% across agents. Results showed stable weather monitoring, effective handling of high-risk traffic plateaus 0.85, and adaptive trust regulation in Safety/Fire scenarios 0.65. Runtime profiling of a 3-agent deployment confirmed scalability, with throughput between 13.8-17.2 requests per second, execution times below 72~ms, and governance delays under 100 ms, analytical projections suggest maintained performance at larger scales. Cross-domain rules ensured safe interoperability, with traffic rerouting permitted only under validated weather conditions. These findings validate SORA-ATMAS as a regulation-aligned, context-aware, and verifiable governance framework that consolidates distributed agent outputs into accountable, real-time decisions, offering a resilient foundation for smart-city management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02344v2">Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      We introduce Traffic-R1, a 3B-parameter foundation model with human-like reasoning for Traffic signal control (TSC), developed via self-exploration and iterative reinforcement of LLM with expert guidance in a simulated traffic environment. Compared with traditional reinforcement learning and recent LLM-based methods, Traffic-R1 offers three main advantages: zero-shot generalization, transferring unchanged to new road networks and out-of-distribution incidents by leveraging internal traffic-control policies and reasoning; a compact 3B-parameter design that supports real-time inference on mobile-class chips for edge deployment; and an explainable TSC process that enables multi-intersection coordination through communication and an asynchronous communication network. Extensive benchmarks show Traffic-R1 outperforms strong baselines and training-intensive RL controllers. In production, the model now manages signals affecting over 55,000 drivers daily, reduces average queue lengths by more than 5%, and halves operator workload. Our model is available at https://huggingface.co/Season998/Traffic-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19299v1">Learning to Make Friends: Coaching LLM Agents toward Emergent Social Ties</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Can large language model (LLM) agents reproduce the complex social dynamics that characterize human online behavior -- shaped by homophily, reciprocity, and social validation -- and what memory and learning mechanisms enable such dynamics to emerge? We present a multi-agent LLM simulation framework in which agents repeatedly interact, evaluate one another, and adapt their behavior through in-context learning accelerated by a coaching signal. To model human social behavior, we design behavioral reward functions that capture core drivers of online engagement, including social interaction, information seeking, self-presentation, coordination, and emotional support. These rewards align agent objectives with empirically observed user motivations, enabling the study of how network structures and group formations emerge from individual decision-making. Our experiments show that coached LLM agents develop stable interaction patterns and form emergent social ties, yielding network structures that mirror properties of real online communities. By combining behavioral rewards with in-context adaptation, our framework establishes a principled testbed for investigating collective dynamics in LLM populations and reveals how artificial agents may approximate or diverge from human-like social behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20214v2">Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at https://github.com/snu-mllab/Q-Palette.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25271v3">RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19264v1">LAPRAD: LLM-Assisted PRotocol Attack Discovery</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 IFIP Networking 2025 Proceedings (Accepted on 05.05.2025)
    </div>
    <details class="paper-abstract">
      With the goal of improving the security of Internet protocols, we seek faster, semi-automatic methods to discover new vulnerabilities in protocols such as DNS, BGP, and others. To this end, we introduce the LLM-Assisted Protocol Attack Discovery (LAPRAD) methodology, enabling security researchers with some DNS knowledge to efficiently uncover vulnerabilities that would otherwise be hard to detect. LAPRAD follows a three-stage process. In the first, we consult an LLM (GPT-o1) that has been trained on a broad corpus of DNS-related sources and previous DDoS attacks to identify potential exploits. In the second stage, a different LLM automatically constructs the corresponding attack configurations using the ReACT approach implemented via LangChain (DNS zone file generation). Finally, in the third stage, we validate the attack's functionality and effectiveness. Using LAPRAD, we uncovered three new DDoS attacks on the DNS protocol and rediscovered two recently reported ones that were not included in the LLM's training data. The first new attack employs a bait-and-switch technique to trick resolvers into caching large, bogus DNSSEC RRSIGs, reducing their serving capacity to as little as 6%. The second exploits large DNSSEC encryption algorithms (RSA-4096) with multiple keys, thereby bypassing a recently implemented default RRSet limit. The third leverages ANY-type responses to produce a similar effect. These variations of a cache-flushing DDoS attack, called SigCacheFlush, circumvent existing patches, severely degrade resolver query capacity, and impact the latest versions of major DNS resolver implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19252v1">LLMartini: Seamless and Interactive Leveraging of Multiple LLMs through Comparison and Composition</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The growing diversity of large language models (LLMs) means users often need to compare and combine outputs from different models to obtain higher-quality or more comprehensive responses. However, switching between separate interfaces and manually integrating outputs is inherently inefficient, leading to a high cognitive burden and fragmented workflows. To address this, we present LLMartini, a novel interactive system that supports seamless comparison, selection, and intuitive cross-model composition tools. The system decomposes responses into semantically aligned segments based on task-specific criteria, automatically merges consensus content, and highlights model differences through color coding while preserving unique contributions. In a user study (N=18), LLMartini significantly outperformed conventional manual methods across all measured metrics, including task completion time, cognitive load, and user satisfaction. Our work highlights the importance of human-centered design in enhancing the efficiency and creativity of multi-LLM interactions and offers practical implications for leveraging the complementary strengths of various language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18179v2">Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 13 pages, 8 figures. Accepted for presentation at the 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although high-performing verifiers can detect reasoning errors, making them reliable requires substantial training. To address these challenges, we introduce a novel inference-time framework, Adaptive Coopetition (AdCo), in which LLM agents utilize an adaptive, UCB-based "coopetition" mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on mathematical reasoning benchmarks, yielding a 20% relative improvement over baselines on the more challenging dataset. Our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided "coopetition" framework enhances reasoning robustness by leveraging both model knowledge diversity and reasoning trace measures, while also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems. Our code is available at: https://github.com/AdCo-Research/adaptive-coopetition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06252v2">ZipLLM: Efficient LLM Storage via Model-Aware Synergistic Data Deduplication and Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Modern model hubs, such as Hugging Face, store tens of petabytes of LLMs, with fine-tuned variants vastly outnumbering base models and dominating storage consumption. Existing storage reduction techniques -- such as deduplication and compression -- are either LLM-oblivious or not compatible with each other, limiting data reduction effectiveness. Our large-scale characterization study across all publicly available Hugging Face LLM repositories reveals several key insights: (1) fine-tuned models within the same family exhibit highly structured, sparse parameter differences suitable for delta compression; (2) bitwise similarity enables LLM family clustering; and (3) tensor-level deduplication is better aligned with model storage workloads, achieving high data reduction with low metadata overhead. Building on these insights, we design BitX, an effective, fast, lossless delta compression algorithm that compresses XORed difference between fine-tuned and base LLMs. We build ZipLLM, a model storage reduction pipeline that unifies tensor-level deduplication and lossless BitX compression. By synergizing deduplication and compression around LLM family clustering, ZipLLM reduces model storage consumption by 54%, over 20% higher than state-of-the-art deduplication and compression approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19225v1">RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become essential for unlocking advanced reasoning capabilities in large language models (LLMs). RL workflows involve interleaving rollout and training stages with fundamentally different resource requirements. Rollout typically dominates overall execution time, yet scales efficiently through multiple independent instances. In contrast, training requires tightly-coupled GPUs with full-mesh communication. Existing RL frameworks fall into two categories: co-located and disaggregated architectures. Co-located ones fail to address this resource tension by forcing both stages to share the same GPUs. Disaggregated architectures, without modifications of well-established RL algorithms, suffer from resource under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances on public clouds and spare capacity in production clusters, present significant cost-saving opportunities for accelerating RL workflows, if efficiently harvested for rollout. In this paper, we present RLBoost, a systematic solution for cost-efficient RL training that harvests preemptible GPU resources. Our key insight is that rollout's stateless and embarrassingly parallel nature aligns perfectly with preemptible and often fragmented resources. To efficiently utilize these resources despite frequent and unpredictable availability changes, RLBoost adopts a hybrid architecture with three key techniques: (1) adaptive rollout offload to dynamically adjust workloads on the reserved (on-demand) cluster, (2) pull-based weight transfer that quickly provisions newly available instances, and (3) token-level response collection and migration for efficient preemption handling and continuous load balancing. Extensive experiments show RLBoost increases training throughput by 1.51x-1.97x while improving cost efficiency by 28%-49% compared to using only on-demand GPU resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18795v2">ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 17 pages, 5 fiugres
    </div>
    <details class="paper-abstract">
      The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19208v1">DiSRouter: Distributed Self-Routing for LLM Selections</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness, its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04961v4">Demystifying Domain-adaptive Post-training for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 EMNLP 2025 (Oral, ARR best paper nomination)
    </div>
    <details class="paper-abstract">
      Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain-adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15831v2">Who's Asking? Investigating Bias Through the Lens of Disability Framed Queries in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) routinely infer users demographic traits from phrasing alone, which can result in biased responses, even when no explicit demographic information is provided. The role of disability cues in shaping these inferences remains largely uncharted. Thus, we present the first systematic audit of disability-conditioned demographic bias across eight state-of-the-art instruction-tuned LLMs ranging from 3B to 72B parameters. Using a balanced template corpus that pairs nine disability categories with six real-world business domains, we prompt each model to predict five demographic attributes - gender, socioeconomic status, education, cultural background, and locality - under both neutral and disability-aware conditions. Across a varied set of prompts, models deliver a definitive demographic guess in up to 97\% of cases, exposing a strong tendency to make arbitrary inferences with no clear justification. Disability context heavily shifts predicted attribute distributions, and domain context can further amplify these deviations. We observe that larger models are simultaneously more sensitive to disability cues and more prone to biased reasoning, indicating that scale alone does not mitigate stereotype amplification. Our findings reveal persistent intersections between ableism and other demographic stereotypes, pinpointing critical blind spots in current alignment strategies. We release our evaluation framework and results to encourage disability-inclusive benchmarking and recommend integrating abstention calibration and counterfactual fine-tuning to curb unwarranted demographic inference. Code and data will be released on acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19178v1">Imbalanced Gradients in RL Post-Training of Multi-Task LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Multi-task post-training of large language models (LLMs) is typically performed by mixing datasets from different tasks and optimizing them jointly. This approach implicitly assumes that all tasks contribute gradients of similar magnitudes; when this assumption fails, optimization becomes biased toward large-gradient tasks. In this paper, however, we show that this assumption fails in RL post-training: certain tasks produce significantly larger gradients, thus biasing updates toward those tasks. Such gradient imbalance would be justified only if larger gradients implied larger learning gains on the tasks (i.e., larger performance improvements) -- but we find this is not true. Large-gradient tasks can achieve similar or even much lower learning gains than small-gradient ones. Further analyses reveal that these gradient imbalances cannot be explained by typical training statistics such as training rewards or advantages, suggesting that they arise from the inherent differences between tasks. This cautions against naive dataset mixing and calls for future work on principled gradient-level corrections for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12499v2">PTFA: An LLM-based Agent that Facilitates Online Consensus Building through Parallel Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Consensus building is inherently challenging due to the diverse opinions held by stakeholders. Effective facilitation is crucial to support the consensus building process and enable efficient group decision making. However, the effectiveness of facilitation is often constrained by human factors such as limited experience and scalability. In this research, we propose a Parallel Thinking-based Facilitation Agent (PTFA) that facilitates online, text-based consensus building processes.The PTFA automatically collects real-time textual input and leverages large language models (LLMs)to perform all six distinct roles of the well-established Six Thinking Hats technique in parallel thinking.To illustrate the potential of the agent, a pilot study was conducted, demonstrating its capabilities in idea generation, emotional probing, and deeper analysis of idea quality. Additionally, future open research challenges such as optimizing scheduling and managing behaviors in divergent phase are identified. Furthermore, a comprehensive dataset that contains not only the conversational content among the participants but also between the participants and the agent is constructed for future study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12631v2">Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 This work has been accepted to DAI 2025
    </div>
    <details class="paper-abstract">
      Balancing performance and efficiency is a central challenge in large language model (LLM) advancement. GPT-5 addresses this with test-time routing, dynamically assigning queries to either an efficient or a high-capacity model during inference. In this work, we present Avengers-Pro, a test-time routing framework that ensembles LLMs of varying capacities and efficiencies, providing a unified solution for all performance-efficiency tradeoffs. The Avengers-Pro embeds and clusters incoming queries, then routes each to the most suitable model based on a performance-efficiency score. Across 6 challenging benchmarks and 8 leading models -- including GPT-5-medium, Gemini-2.5-pro, and Claude-opus-4.1 -- Avengers-Pro achieves state-of-the-art results: by varying a performance-efficiency trade-off parameter, it can surpass the strongest single model (GPT-5-medium) by +7% in average accuracy. Moreover, it can match the average accuracy of the strongest single model at 27% lower cost, and reach ~90% of that performance at 63% lower cost. Last but not least, it achieves a Pareto frontier, consistently yielding the highest accuracy for any given cost, and the lowest cost for any given accuracy, among all single models. Code is available at https://github.com/ZhangYiqun018/AvengersPro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19172v1">When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18279v2">Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted to EMNLP 2025 Findings ("Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs")
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18527v2">LLMs as Sparse Retrievers:A Framework for First-Stage Product Search</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Product search is a crucial component of modern e-commerce platforms, with billions of user queries every day. In product search systems, first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval is particularly attractive in this context due to its interpretability and storage efficiency. However, sparse retrieval methods suffer from severe vocabulary mismatch issues, leading to suboptimal performance in product search scenarios. With their potential for semantic analysis, large language models (LLMs) offer a promising avenue for mitigating vocabulary mismatch issues and thereby improving retrieval quality. Directly applying LLMs to sparse retrieval in product search exposes two key challenges:(1)Queries and product titles are typically short and highly susceptible to LLM-induced hallucinations, such as generating irrelevant expansion terms or underweighting critical literal terms like brand names and model numbers;(2)The large vocabulary space of LLMs leads to difficulty in initializing training effectively, making it challenging to learn meaningful sparse representations in such ultra-high-dimensional spaces.To address these challenges, we propose PROSPER, a framework for PROduct search leveraging LLMs as SParsE Retrievers. PROSPER incorporates: (1)A literal residual network that alleviates hallucination in lexical expansion by reinforcing underweighted literal terms through a residual compensation mechanism; and (2)A lexical focusing window that facilitates effective training initialization via a coarse-to-fine sparsification strategy.Extensive offline and online experiments show that PROSPER significantly outperforms sparse baselines and achieves recall performance comparable to advanced dense retrievers, while also achieving revenue increments online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05735v4">Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 NeurIPS Camera-Ready Version. Code available at: https://github.com/Graph-COM/Knowledge_Unlearning
    </div>
    <details class="paper-abstract">
      Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07897v3">LongCodeBench: Evaluating Coding LLMs at 1M Context Windows</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5. The LCB dataset is available publicly at https://huggingface.co/datasets/Steefano/LCB and the codebase to replicate the work on this paper at https://github.com/Zteefano/long-code-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20075v1">LLMs can hide text in other text of the same length.ipynb</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 21 pages, main paper 9 pages
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14522v2">Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04510v2">Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We propose Heterogeneous Swarms, an algorithm to design multi-LLM systems by jointly optimizing model roles and weights. We represent multi-LLM systems as directed acyclic graphs (DAGs) of LLMs with topological message passing for collaborative generation. Given a pool of LLM experts and a utility function, Heterogeneous Swarms employs two iterative steps: role-step and weight-step. For role-step, we interpret model roles as learning a DAG that specifies the flow of inputs and outputs between LLMs. Starting from a swarm of random continuous adjacency matrices, we decode them into discrete DAGs, call the LLMs in topological order, evaluate on the utility function (e.g. accuracy on a task), and optimize the adjacency matrices with particle swarm optimization based on the utility score. For weight-step, we assess the contribution of individual LLMs in the multi-LLM systems and optimize model weights with swarm intelligence. We propose JFK-score to quantify the individual contribution of each LLM in the best-found DAG of the role-step, then optimize model weights with particle swarm optimization based on the JFK-score. Experiments demonstrate that Heterogeneous Swarms outperforms 15 role- and/or weight-based baselines by 18.5% on average across 12 tasks. Further analysis reveals that Heterogeneous Swarms discovers multi-LLM systems with heterogeneous model roles and substantial collaborative gains, and benefits from the diversity of language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00418v2">Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and NVIDIA Data Center GPUs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 8 pages, 3 tables
    </div>
    <details class="paper-abstract">
      This study presents a benchmarking analysis of the Qualcomm Cloud AI 100 Ultra (QAic) accelerator for large language model (LLM) inference, evaluating its energy efficiency (throughput per watt), performance, and hardware scalability against NVIDIA A100 GPUs (in 4x and 8x configurations) within the National Research Platform (NRP) ecosystem. A total of 12 open-source LLMs, ranging from 124 million to 70 billion parameters, are served using the vLLM framework. Our analysis reveals that QAic achieves competitive energy efficiency with advantages on specific models while enabling more granular hardware allocation: some 70B models operate on as few as 1 QAic card versus 8 A100 GPUs required, with 20x lower power consumption (148W vs 2,983W). For smaller models, single QAic devices achieve up to 35x lower power consumption compared to our 4-GPU A100 configuration (36W vs 1,246W). The findings offer insights into the potential of the Qualcomm Cloud AI 100 Ultra for energy-constrained and resource-efficient HPC deployments within the National Research Platform (NRP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20064v1">Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Speculative decoding is widely used in accelerating large language model (LLM) inference. In this work, we focus on the online draft model selection problem in speculative decoding. We design an algorithm that provably competes with the best draft model in hindsight for each query in terms of either the token acceptance probability or expected acceptance length. In particular, we show that we can accurately evaluate all draft models, instead of only the chosen model without incurring additional queries to the target model, which allows us to improve exponentially over the existing bandit-based approach as the number of draft models increases. Our approach is generically applicable with any speculative decoding methods (single draft, multi-drafts and draft-trees). Moreover, we design system-efficient versions of online learners and demonstrate that the overhead in computation and latency can be substantially reduced. We conduct extensive experiments on open-source LLMs and diverse datasets, demonstrating that our methods substantially outperform the state-of-the-art EAGLE3 and the BanditSpec baseline in a variety of domains where specialized domain-expert drafters are available, especially when long reasoning chains are required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20039v1">Beyond One-Way Influence: Bidirectional Opinion Dynamics in Multi-Turn Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered chatbots are increasingly used for opinion exploration. Prior research examined how LLMs alter user views, yet little work extended beyond one-way influence to address how user input can affect LLM responses and how such bi-directional influence manifests throughout the multi-turn conversations. This study investigates this dynamic through 50 controversial-topic discussions with participants (N=266) across three conditions: static statements, standard chatbot, and personalized chatbot. Results show that human opinions barely shifted, while LLM outputs changed more substantially, narrowing the gap between human and LLM stance. Personalization amplified these shifts in both directions compared to the standard setting. Analysis of multi-turn conversations further revealed that exchanges involving participants' personal stories were most likely to trigger stance changes for both humans and LLMs. Our work highlights the risk of over-alignment in human-LLM interaction and the need for careful design of personalized chatbots to more thoughtfully and stably align with users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20036v1">ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Preprint under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents rely on external tools to solve complex tasks, but real-world toolsets often contain redundant tools with overlapping names and descriptions, introducing ambiguity and reducing selection accuracy. LLMs also face strict input context limits, preventing efficient consideration of large toolsets. To address these challenges, we propose ToolScope, which includes: (1) ToolScopeMerger with Auto-Correction to automatically audit and fix tool merges, reducing redundancy, and (2) ToolScopeRetriever to rank and select only the most relevant tools for each query, compressing toolsets to fit within context limits without sacrificing accuracy. Evaluations on three state-of-the-art LLMs and three open-source tool-use benchmarks show gains of 8.38% to 38.6% in tool selection accuracy, demonstrating ToolScope's effectiveness in enhancing LLM tool use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18882v3">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20001v1">Beyond MedQA: Towards Real-world Clinical Decision Making in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for clinical use. They are often evaluated using datasets such as MedQA. However, Many medical datasets, such as MedQA, rely on simplified Question-Answering (Q\A) that underrepresents real-world clinical decision-making. Based on this, we propose a unifying paradigm that characterizes clinical decision-making tasks along two dimensions: Clinical Backgrounds and Clinical Questions. As the background and questions approach the real clinical environment, the difficulty increases. We summarize the settings of existing datasets and benchmarks along two dimensions. Then we review methods to address clinical decision-making, including training-time and test-time techniques, and summarize when they help. Next, we extend evaluation beyond accuracy to include efficiency, explainability. Finally, we highlight open challenges. Our paradigm clarifies assumptions, standardizes comparisons, and guides the development of clinically meaningful LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08872v2">GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at https://github.com/ulab-uiuc/GTAlign .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19988v1">LLM-Augmented Symbolic NLU System for More Reliable Continuous Causal Statement Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 18 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Despite the broad applicability of large language models (LLMs), their reliance on probabilistic inference makes them vulnerable to errors such as hallucination in generated facts and inconsistent output structure in natural language understanding (NLU) tasks. By contrast, symbolic NLU systems provide interpretable understanding grounded in curated lexicons, semantic resources, and syntactic & semantic interpretation rules. They produce relational representations that can be used for accurate reasoning and planning, as well as incremental debuggable learning. However, symbolic NLU systems tend to be more limited in coverage than LLMs and require scarce knowledge representation and linguistics skills to extend and maintain. This paper explores a hybrid approach that integrates the broad-coverage language processing of LLMs with the symbolic NLU capabilities of producing structured relational representations to hopefully get the best of both approaches. We use LLMs for rephrasing and text simplification, to provide broad coverage, and as a source of information to fill in knowledge gaps more automatically. We use symbolic NLU to produce representations that can be used for reasoning and for incremental learning. We evaluate this approach on the task of extracting and interpreting quantities and causal laws from commonsense science texts, along with symbolic- and LLM-only pipelines. Our results suggest that our hybrid method works significantly better than the symbolic-only pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19986v1">Automating Iconclass: LLMs and RAG for Large-Scale Classification of Religious Woodcuts</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 29 pages, 7 figures. First presented at the "Digital Humanities and Artificial Intelligence" conference at the University of Reading on 17 June 2024
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for classifying early modern religious images by using Large Language Models (LLMs) and vector databases in combination with Retrieval-Augmented Generation (RAG). The approach leverages the full-page context of book illustrations from the Holy Roman Empire, allowing the LLM to generate detailed descriptions that incorporate both visual and textual elements. These descriptions are then matched to relevant Iconclass codes through a hybrid vector search. This method achieves 87% and 92% precision at five and four levels of classification, significantly outperforming traditional image and keyword-based searches. By employing full-page descriptions and RAG, the system enhances classification accuracy, offering a powerful tool for large-scale analysis of early modern visual archives. This interdisciplinary approach demonstrates the growing potential of LLMs and RAG in advancing research within art history and digital humanities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19886v1">An Expert-grounded benchmark of General Purpose LLMs in LCA</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Purpose: Artificial intelligence (AI), and in particular large language models (LLMs), are increasingly being explored as tools to support life cycle assessment (LCA). While demonstrations exist across environmental and social domains, systematic evidence on their reliability, robustness, and usability remains limited. This study provides the first expert-grounded benchmark of LLMs in LCA, addressing the absence of standardized evaluation frameworks in a field where no clear ground truth or consensus protocols exist. Methods: We evaluated eleven general-purpose LLMs, spanning both commercial and open-source families, across 22 LCA-related tasks. Seventeen experienced practitioners reviewed model outputs against criteria directly relevant to LCA practice, including scientific accuracy, explanation quality, robustness, verifiability, and adherence to instructions. We collected 168 expert reviews. Results: Experts judged 37% of responses to contain inaccurate or misleading information. Ratings of accuracy and quality of explanation were generally rated average or good on many models even smaller models, and format adherence was generally rated favourably. Hallucination rates varied significantly, with some models producing hallucinated citations at rates of up to 40%. There was no clear-cut distinction between ratings on open-weight versus closed-weight LLMs, with open-weight models outperforming or competing on par with closed-weight models on criteria such as accuracy and quality of explanation. Conclusion: These findings highlight the risks of applying LLMs na\"ively in LCA, such as when LLMs are treated as free-form oracles, while also showing benefits especially around quality of explanation and alleviating labour intensiveness of simple tasks. The use of general-purpose LLMs without grounding mechanisms presents ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21015v2">Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Training effective dense retrieval models typically relies on hard negative (HN) examples mined from large document corpora using methods such as BM25 or cross-encoders (CE), which require full corpus access. We propose a corpus-free alternative: an end-to-end pipeline where a Large Language Model (LLM) first generates a query from a passage and then produces a hard negative example using only the generated query text. Our dataset comprises 7,250 arXiv abstracts spanning diverse domains including mathematics, physics, computer science, and related fields, serving as positive passages for query generation. We evaluate two fine-tuning configurations of DistilBERT for dense retrieval; one using LLM-generated hard negatives conditioned solely on the query, and another using negatives generated with both the query and its positive document as context. Compared to traditional corpus-based mining methods {LLM Query $\rightarrow$ BM25 HN and LLM Query $\rightarrow$ CE HN on multiple BEIR benchmark datasets, our all-LLM pipeline outperforms strong lexical mining baselines and achieves performance comparable to cross-encoder-based methods, demonstrating the potential of corpus-free hard negative generation for retrieval model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v5">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18314v1">Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17173v2">Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted to the NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      We study a web-deployed, tool-augmented LLM health coach with real users. In a pilot with seven users (280 rated turns), offline policy evaluation (OPE) over factorized decision heads (Tool/Style) shows that a uniform heavy-tool policy raises average value on logs but harms specific subgroups, most notably low-health-literacy/high-self-efficacy users. A lightweight simulator with hidden archetypes further shows that adding a small early information-gain bonus reliably shortens trait identification and improves goal success and pass@3. Together, these early findings indicate an evaluation-first path to personalization: freeze the generator, learn subgroup-aware decision heads on typed rewards (objective tool outcomes and satisfaction), and always report per-archetype metrics to surface subgroup harms that averages obscure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17727v2">Enabling Fine-Grained Operating Points for Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Under review at ICLR 2026. 36 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Black-box Large Language Models (LLMs) provide practical and accessible alternatives to other machine learning methods, as they require minimal labeled data and machine learning expertise to develop solutions for various decision making problems. However, for applications that need operating with constraints on specific metrics (e.g., precision $\geq$ 95%), decision making with black-box LLMs remains unfavorable, due to their low numerical output cardinalities. This results in limited control over their operating points, preventing fine-grained adjustment of their decision making behavior. In this paper, we study using black-box LLMs as classifiers, focusing on efficiently improving their operational granularity without performance loss. Specifically, we first investigate the reasons behind their low-cardinality numerical outputs and show that they are biased towards generating rounded but informative verbalized probabilities. Then, we experiment with standard prompt engineering, uncertainty estimation and confidence elicitation techniques, and observe that they do not effectively improve operational granularity without sacrificing performance or increasing inference cost. Finally, we propose efficient approaches to significantly increase the number and diversity of available operating points. Our proposed approaches provide finer-grained operating points and achieve comparable to or better performance than the benchmark methods across 11 datasets and 3 LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10255v2">When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 2nd version update to Jun.2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve, their integration with 3D spatial data (3D-LLMs) has seen rapid progress, offering unprecedented capabilities for understanding and interacting with physical spaces. This survey provides a comprehensive overview of the methodologies enabling LLMs to process, understand, and generate 3D data. Highlighting the unique advantages of LLMs, such as in-context learning, step-by-step reasoning, open-vocabulary capabilities, and extensive world knowledge, we underscore their potential to significantly advance spatial comprehension and interaction within embodied Artificial Intelligence (AI) systems. Our investigation spans various 3D data representations, from point clouds to Neural Radiance Fields (NeRFs). It examines their integration with LLMs for tasks such as 3D scene understanding, captioning, question-answering, and dialogue, as well as LLM-based agents for spatial reasoning, planning, and navigation. The paper also includes a brief review of other methods that integrate 3D and language. The meta-analysis presented in this paper reveals significant progress yet underscores the necessity for novel approaches to harness the full potential of 3D-LLMs. Hence, with this paper, we aim to chart a course for future research that explores and expands the capabilities of 3D-LLMs in understanding and interacting with the complex 3D world. To support this survey, we have established a project page where papers related to our topic are organized and listed: https://github.com/ActiveVisionLab/Awesome-LLM-3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14162v2">FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 6 pages, 2 figures, accepted at CIKM 2025 FinAI Workshop
    </div>
    <details class="paper-abstract">
      We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18279v1">Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted to EMNLP 2025 Findings. Previously titled "Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs"
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18277v1">Enhancing Hotel Recommendations with AI: LLM-Based Review Summarization and Query-Driven Insights</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The increasing number of data a booking platform such as Booking.com and AirBnB offers make it challenging for interested parties to browse through the available accommodations and analyze reviews in an efficient way. Efforts have been made from the booking platform providers to utilize recommender systems in an effort to enable the user to filter the results by factors such as stars, amenities, cost but most valuable insights can be provided by the unstructured text-based reviews. Going through these reviews one-by-one requires a substantial amount of time to be devoted while a respectable percentage of the reviews won't provide to the user what they are actually looking for. This research publication explores how Large Language Models (LLMs) can enhance short rental apartments recommendations by summarizing and mining key insights from user reviews. The web application presented in this paper, named "instaGuide", automates the procedure of isolating the text-based user reviews from a property on the Booking.com platform, synthesizing the summary of the reviews, and enabling the user to query specific aspects of the property in an effort to gain feedback on their personal questions/criteria. During the development of the instaGuide tool, numerous LLM models were evaluated based on accuracy, cost, and response quality. The results suggest that the LLM-powered summarization reduces significantly the amount of time the users need to devote on their search for the right short rental apartment, improving the overall decision-making procedure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09423v2">Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The Object Goal Navigation (ObjectNav) task challenges agents to locate a specified object in an unseen environment by imagining unobserved regions of the scene. Prior approaches rely on deterministic and discriminative models to complete semantic maps, overlooking the inherent uncertainty in indoor layouts and limiting their ability to generalize to unseen environments. In this work, we propose GOAL, a generative flow-based framework that models the semantic distribution of indoor environments by bridging observed regions with LLM-enriched full-scene semantic maps. During training, spatial priors inferred from large language models (LLMs) are encoded as two-dimensional Gaussian fields and injected into target maps, distilling rich contextual knowledge into the flow model and enabling more generalizable completions. Extensive experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D and Gibson, and shows strong generalization in transfer settings to HM3D. Codes and pretrained models are available at https://github.com/Badi-Li/GOAL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19856v1">Model Context Contracts - MCP-Enabled Framework to Integrate LLMs With Blockchain Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      In recent years, blockchain has experienced widespread adoption across various industries, becoming integral to numerous enterprise applications. Concurrently, the rise of generative AI and LLMs has transformed human-computer interactions, offering advanced capabilities in understanding and generating human-like text. The introduction of the MCP has further enhanced AI integration by standardizing communication between AI systems and external data sources. Despite these advancements, there is still no standardized method for seamlessly integrating LLM applications and blockchain. To address this concern, we propose "MCC: Model Context Contracts" a novel framework that enables LLMs to interact directly with blockchain smart contracts through MCP-like protocol. This integration allows AI agents to invoke blockchain smart contracts, facilitating more dynamic and context-aware interactions between users and blockchain networks. Essentially, it empowers users to interact with blockchain systems and perform transactions using queries in natural language. Within this proposed architecture, blockchain smart contracts can function as intelligent agents capable of recognizing user input in natural language and executing the corresponding transactions. To ensure that the LLM accurately interprets natural language inputs and maps them to the appropriate MCP functions, the LLM was fine-tuned using a custom dataset comprising user inputs paired with their corresponding MCP server functions. This fine-tuning process significantly improved the platform's performance and accuracy. To validate the effectiveness of MCC, we have developed an end-to-end prototype implemented on the Rahasak blockchain with the fine-tuned Llama-4 LLM. To the best of our knowledge, this research represents the first approach to using the concept of Model Context Protocol to integrate LLMs with blockchain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19850v1">Prompt Decorators: A Declarative and Composable Syntax for Reasoning, Formatting, and Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are central to reasoning, writing, and decision-support workflows, yet users lack consistent control over how they reason and express outputs. Conventional prompt engineering relies on verbose natural-language instructions, limiting reproducibility, modularity, and interpretability. This paper introduces Prompt Decorators, a declarative, composable syntax that governs LLM behavior through compact control tokens such as +++Reasoning, +++Tone(style=formal), and +++Import(topic="Systems Thinking"). Each decorator modifies a behavioral dimension, such as reasoning style, structure, or tone, without changing task content. The framework formalizes twenty core decorators organized into two functional families (Cognitive & Generative and Expressive & Systemic), each further decomposed into subcategories that govern reasoning, interaction, expression, and session-control. It defines a unified syntax, scoping model, and deterministic processing pipeline enabling predictable and auditable behavior composition. By decoupling task intent from execution behavior, Prompt Decorators create a reusable and interpretable interface for prompt design. Illustrative use cases demonstrate improved reasoning transparency, reduced prompt complexity, and standardized model behavior across domains. The paper concludes with implications for interoperability, behavioral consistency, and the development of declarative interfaces for scalable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23804v3">DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders. To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18250v1">ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Data quality plays a critical role in enhancing supervised fine-tuning (SFT) for large language models (LLMs), and token-level data selection has emerged as a promising direction for its fine-grained nature. Despite their strong empirical performance, existing token-level selection methods share two key limitations: (1) requiring training or accessing an additional reference model, and (2) relying solely on loss information for token selection, which cannot well preserve semantically important tokens that are not favored by loss-based metrics. To address these challenges, we propose ssToken, a Self-modulated and Semantic-aware Token Selection approach. ssToken leverages readily accessible history models to compute the per-token loss difference with the current model, which serves as a self-modulated signal that enables the model to adaptively select tokens along its optimization trajectory, rather than relying on excess loss from an offline-trained reference model as in prior works. We further introduce a semantic-aware, attention-based token importance estimation metric, orthogonal to loss-based selection and providing complementary semantic information for more effective filtering. Extensive experiments across different model families and scales demonstrate that both self-modulated selection and semantic-aware selection alone outperform full-data fine-tuning, while their integration--ssToken--achieves synergistic gains and further surpasses prior token-level selection methods, delivering performance improvements while maintaining training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18245v1">Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 27 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Scaling the number of parameters and the size of training data has proven to be an effective strategy for improving large language model (LLM) performance. Yet, as these models grow increasingly powerful and widely deployed, the cost of inference has become a pressing concern. Despite its importance, the trade-off between model accuracy and inference efficiency remains underexplored. In this work, we examine how key architectural factors, hidden size, the allocation of parameters between MLP and attention (mlp-to-attention ratio), and grouped-query attention (GQA), influence both inference cost and accuracy. We introduce a conditional scaling law that augments the Chinchilla framework with architectural information, along with a search framework for identifying architectures that are simultaneously inference-efficient and accurate. To validate our approach, we train more than 200 models spanning 80M to 3B parameters and 8B to 100B training tokens, and fit the proposed conditional scaling law. Our results show that the conditional scaling law reliably predicts optimal architectural choices and that the resulting models outperform existing open-source baselines. Under the same training budget, optimized architectures achieve up to 2.1% higher accuracy and 42% greater inference throughput compared to LLaMA-3.2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16456v2">GPO: Learning from Critical Steps to Improve LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18228v1">Towards Fast LLM Fine-tuning through Zeroth-Order Optimization with Projected Gradient-Aligned Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using zeroth-order (ZO) optimization has emerged as a promising alternative to traditional gradient-based methods due to its reduced memory footprint requirement. However, existing ZO methods suffer from high variance in gradient estimation, leading to slow convergence and suboptimal performance on large-scale models. In this work, we propose P-GAP, a fast LLM fine-tuning approach through zeroth-order optimization with Projected Gradient-Aligned Perturbations. Specifically, we first estimate a low-dimensional gradient space and then align perturbations in projected gradients' direction within the space. This approach enables reduced the number of perturbed parameters and decreased variance, therefore accelerated convergence for LLM fine-tuning. Experiments on LLMs show that P-GAP consistently surpasses the baselines, achieving up to 6% increase in accuracy on classification tasks and up to 12% higher accuracy on generation tasks, with up to about 81% less training iterations and 70% less GPU hours. These results demonstrate that P-GAP enables fast, scalable, and resource-efficient ZO LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14253v2">Towards Agentic Self-Learning LLMs in Search Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at https://github.com/forangel2014/Towards-Agentic-Self-Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09657v4">Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) yet often fails to maintain comparable performance. Local pruning performs efficient layer-by-layer compression but ignores global topology. Although global pruning aims to identify an optimal sparse model, intuitive methods typically adopt a two-stage paradigm that first evaluates substructure saliency and then applies global pruning, which ignores inter-structure dependencies and fails to achieve end-to-end optimization. To address these limitations, we propose T\'yr-the-Pruner, an efficient end-to-end search-based global structural pruning framework. This framework constructs a supernet by repeatedly applying local pruning across a range of sparsity ratios to each layer in an LLM, with the core goal of determining the optimal sparsity distribution under a target overall sparsity ratio. Concretely, we introduce an effective local pruning and an expectation error accumulation approach to improve supernet construction. Furthermore, we employ an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to ensure efficient search convergence. Experimental results show that T\'yr-the-Pruner achieves state-of-the-art structural pruning, retaining 97% of the dense model's performance while removing a challenging 50% of Llama-3.1-70B's parameters. Code will be available at https://github.com/AMD-AGI/Tyr-the-Pruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23967v2">HiPO: Hybrid Policy Optimization for Dynamic Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on Chain-of-Thought (CoT) reasoning to improve accuracy on complex tasks. However, always generating lengthy reasoning traces is inefficient, leading to excessive token usage and higher inference costs. This paper introduces the Hybrid Policy Optimization (i.e., HiPO), a framework for adaptive reasoning control that enables LLMs to selectively decide when to engage in detailed reasoning (Think-on) and when to respond directly (Think-off). Specifically, HiPO combines a hybrid data pipelineproviding paired Think-on and Think-off responseswith a hybrid reinforcement learning reward system that balances accuracy and efficiency while avoiding over-reliance on detailed reasoning. Experiments across mathematics and coding benchmarks demonstrate that HiPO can substantially reduce token length while maintaining or improving accuracy. Finally, we hope HiPO a can be a principled approach for efficient adaptive reasoning, advancing the deployment of reasoning-oriented LLMs in real-world, resource-sensitive settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18196v1">Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly used as evaluators in various applications, but the reliability of the outcomes remains a challenge. One such challenge is using LLMs-as-judges for direct assessment, i.e., assigning scores from a specified range without any references. We first show that this challenge stems from LLM judge outputs being associated with score range bias, i.e., LLM judge outputs are highly sensitive to pre-defined score ranges, preventing the search for optimal score ranges. We also show that similar biases exist among models from the same family. We then mitigate this bias through contrastive decoding, achieving up to 11.3% relative improvement on average in Spearman correlation with human judgments across different score ranges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21837v2">Semantic Agreement Enables Efficient Open-Ended LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Industry Track
    </div>
    <details class="paper-abstract">
      Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12112v6">Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to design reward functions based on human preferences in Reinforcement Learning (RL). We focus on LLM-designed rewards for Restless Multi-Armed Bandits, a framework for allocating limited resources among agents. In applications such as public health, this approach empowers grassroots health workers to tailor automated allocation decisions to community needs. In the presence of multiple agents, altering the reward function based on human preferences can impact subpopulations very differently, leading to complex tradeoffs and a multi-objective resource allocation problem. We are the first to present a principled method termed Social Choice Language Model for dealing with these tradeoffs for LLM-designed rewards for multiagent planners in general and restless bandits in particular. The novel part of our model is a transparent and configurable selection component, called an adjudicator, external to the LLM that controls complex tradeoffs via a user-selected social welfare function. Our experiments demonstrate that our model reliably selects more effective, aligned, and balanced reward functions compared to purely LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18179v1">Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although high-performing verifiers can detect reasoning errors, making them reliable requires substantial training. To address these challenges, we introduce a novel inference-time framework, Adaptive Coopetition (AdCo), in which LLM agents utilize an adaptive, UCB-based "coopetition" mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on mathematical reasoning benchmarks, yielding a 20% relative improvement over baselines on the more challenging dataset. Our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided "coopetition" framework enhances reasoning robustness by leveraging both model knowledge diversity and reasoning trace measures, while also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems. Our code is available at: https://github.com/AdCo-Research/adaptive-coopetition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06313v3">ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11857v3">LLMBridge: Reducing Costs to Access LLMs in a Prompt-Centric Internet</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Today's Internet infrastructure is centered around content retrieval over HTTP, with middleboxes (e.g., HTTP proxies) playing a crucial role in performance, security, and cost-effectiveness. We envision a future where Internet communication will be dominated by "prompts" sent to generative AI models. For this, we will need proxies that provide similar functions to HTTP proxies (e.g., caching, routing, compression) while dealing with unique challenges and opportunities of prompt-based communication. As a first step toward supporting prompt-based communication, we present LLMBridge, an LLM proxy designed for cost-conscious users, such as those in developing regions and education (e.g., students, instructors). LLMBridge supports three key optimizations: model selection (routing prompts to the most suitable model), context management (intelligently reducing the amount of context), and semantic caching (serving prompts using local models and vector databases). These optimizations introduce trade-offs between cost and quality, which applications navigate through a high-level, bidirectional interface. As case studies, we deploy LLMBridge in two cost-sensitive settings: a WhatsApp-based Q&A service and a university classroom environment. The WhatsApp service has been live for over twelve months, serving 100+ users and handling more than 14.7K requests. In parallel, we exposed LLMBridge to students across three computer science courses over a semester, where it supported diverse LLM-powered applications - such as reasoning agents and chatbots - and handled an average of 500 requests per day. We report on deployment experiences across both settings and use the collected workloads to benchmark the effectiveness of various cost-optimization strategies, analyzing their trade-offs in cost, latency, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19107v1">When Your AI Agent Succumbs to Peer-Pressure: Studying Opinion-Change Dynamics of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We investigate how peer pressure influences the opinions of Large Language Model (LLM) agents across a spectrum of cognitive commitments by embedding them in social networks where they update opinions based on peer perspectives. Our findings reveal key departures from traditional conformity assumptions. First, agents follow a sigmoid curve: stable at low pressure, shifting sharply at threshold, and saturating at high. Second, conformity thresholds vary by model: Gemini 1.5 Flash requires over 70% peer disagreement to flip, whereas ChatGPT-4o-mini shifts with a dissenting minority. Third, we uncover a fundamental "persuasion asymmetry," where shifting an opinion from affirmative-to-negative requires a different cognitive effort than the reverse. This asymmetry results in a "dual cognitive hierarchy": the stability of cognitive constructs inverts based on the direction of persuasion. For instance, affirmatively-held core values are robust against opposition but easily adopted from a negative stance, a pattern that inverts for other constructs like attitudes. These dynamics echoing complex human biases like negativity bias, prove robust across different topics and discursive frames (moral, economic, sociotropic). This research introduces a novel framework for auditing the emergent socio-cognitive behaviors of multi-agent AI systems, demonstrating their decision-making is governed by a fluid, context-dependent architecture, not a static logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19099v1">What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 8 pages (main text) + 4 pages (appendix), 4 figures
    </div>
    <details class="paper-abstract">
      Curriculum learning (CL) - ordering training data from easy to hard - has become a popular strategy for improving reasoning in large language models (LLMs). Yet prior work employs disparate difficulty metrics and training setups, leaving open fundamental questions: When does curriculum help? Which direction - forward or reverse - is better? And does the answer depend on what we measure? We address these questions through a unified offline evaluation framework that decomposes curriculum difficulty into five complementary dimensions: Problem Difficulty, Model Surprisal, Confidence Margin, Predictive Uncertainty, and Decision Variability. Through controlled post-training experiments on mathematical reasoning benchmarks with Llama3.1-8B, Mistral-7B, and Gemma3-4B, we find that (i) no curriculum strategy dominates universally - the relative effectiveness of forward versus reverse CL depends jointly on model capability and task complexity; (ii) even within a single metric, samples at different difficulty levels produce distinct gains depending on task demands; and (iii) task-aligned curricula focus on shaping the model's final representations and generalization, whereas inner-state curricula modulate internal states such as confidence and uncertainty. Our findings challenge the notion of a universal curriculum strategy and offer actionable guidance across model and task regimes, with some metrics indicating that prioritizing decision-uncertain samples can further enhance learning outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19060v1">PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 24 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
    </div>
    <details class="paper-abstract">
      While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03502v2">ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 47 pages, 15 figures. Dataset available at Zenodo: https://doi.org/10.5281/zenodo.17249602 Codebase available at GitHub: https://github.com/alikhairallah/ALHD-Benchmarking
    </div>
    <details class="paper-abstract">
      We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19032v1">When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) for mental health support is challenging due to the emotionally and cognitively complex nature of therapeutic dialogue. Existing benchmarks are limited in scale, reliability, often relying on synthetic or social media data, and lack frameworks to assess when automated judges can be trusted. To address the need for large-scale dialogue datasets and judge reliability assessment, we introduce two benchmarks that provide a framework for generation and evaluation. MentalBench-100k consolidates 10,000 one-turn conversations from three real scenarios datasets, each paired with nine LLM-generated responses, yielding 100,000 response pairs. MentalAlign-70k}reframes evaluation by comparing four high-performing LLM judges with human experts across 70,000 ratings on seven attributes, grouped into Cognitive Support Score (CSS) and Affective Resonance Score (ARS). We then employ the Affective Cognitive Agreement Framework, a statistical methodology using intraclass correlation coefficients (ICC) with confidence intervals to quantify agreement, consistency, and bias between LLM judges and human experts. Our analysis reveals systematic inflation by LLM judges, strong reliability for cognitive attributes such as guidance and informativeness, reduced precision for empathy, and some unreliability in safety and relevance. Our contributions establish new methodological and empirical foundations for reliable, large-scale evaluation of LLMs in mental health. We release the benchmarks and codes at: https://github.com/abeerbadawi/MentalBench/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19028v1">Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19025v1">FlexiDataGen: An Adaptive LLM Framework for Dynamic Semantic Dataset Generation in Sensitive Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Dataset availability and quality remain critical challenges in machine learning, especially in domains where data are scarce, expensive to acquire, or constrained by privacy regulations. Fields such as healthcare, biomedical research, and cybersecurity frequently encounter high data acquisition costs, limited access to annotated data, and the rarity or sensitivity of key events. These issues-collectively referred to as the dataset challenge-hinder the development of accurate and generalizable machine learning models in such high-stakes domains. To address this, we introduce FlexiDataGen, an adaptive large language model (LLM) framework designed for dynamic semantic dataset generation in sensitive domains. FlexiDataGen autonomously synthesizes rich, semantically coherent, and linguistically diverse datasets tailored to specialized fields. The framework integrates four core components: (1) syntactic-semantic analysis, (2) retrieval-augmented generation, (3) dynamic element injection, and (4) iterative paraphrasing with semantic validation. Together, these components ensure the generation of high-quality, domain-relevant data. Experimental results show that FlexiDataGen effectively alleviates data shortages and annotation bottlenecks, enabling scalable and accurate machine learning model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19006v1">XGen-Q: An Explainable Domain-Adaptive LLM Framework with Retrieval-Augmented Generation for Software Security</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Generative AI and large language models (LLMs) have shown strong capabilities in code understanding, but their use in cybersecurity, particularly for malware detection and analysis, remains limited. Existing detection systems often fail to generalize to obfuscated or previously unseen threats, underscoring the need for more adaptable and explainable models. To address this challenge, we introduce XGen-Q, a domain-adapted LLM built on the Qwen-Coder architecture and pretrained on a large-scale corpus of over one million malware samples, spanning both source and assembly code. XGen-Q uses a multi-stage prompt strategy combined with retrieval-augmented generation (RAG) to deliver reliable malware identification and detailed forensic reporting, even in the presence of complex code obfuscation. To further enhance generalization, we design a training pipeline that systematically exposes the model to diverse obfuscation patterns. Experimental results show that XGen-Q achieves significantly lower perplexity than competitive baselines and exhibits strong performance on novel malware samples, demonstrating the promise of LLM-based approaches for interpretable and robust malware analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19005v1">Dynamic Evaluation for Oversensitivity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 EMNLP-Findings 2025
    </div>
    <details class="paper-abstract">
      Oversensitivity occurs when language models defensively reject prompts that are actually benign. This behavior not only disrupts user interactions but also obscures the boundary between harmful and harmless content. Existing benchmarks rely on static datasets that degrade overtime as models evolve, leading to data contamination and diminished evaluative power. To address this, we develop a framework that dynamically generates model-specific challenging datasets, capturing emerging defensive patterns and aligning with each model's unique behavior. Building on this approach, we construct OVERBENCH, a benchmark that aggregates these datasets across diverse LLM families, encompassing 450,000 samples from 25 models. OVERBENCH provides a dynamic and evolving perspective on oversensitivity, allowing for continuous monitoring of defensive triggers as models advance, highlighting vulnerabilities that static datasets overlook.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16916v2">SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer promising capabilities for tackling complex reasoning tasks, including optimization problems. However, existing methods either rely on prompt engineering, which leads to poor generalization across problem types, or require costly supervised training. We introduce SolverLLM, a training-free framework that leverages test-time scaling to solve diverse optimization problems. Rather than solving directly, SolverLLM generates mathematical formulations and translates them into solver-ready code, guided by a novel Monte Carlo Tree Search (MCTS) strategy. To enhance the search process, we modify classical MCTS with (1) dynamic expansion for adaptive formulation generation, (2) prompt backpropagation to guide exploration via outcome-driven feedback, and (3) uncertainty backpropagation to incorporate reward reliability into decision-making. Experiments on six standard benchmark datasets demonstrate that SolverLLM outperforms both prompt-based and learning-based baselines, achieving strong generalization without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18932v1">Evaluating LLM Story Generation through Large-scale Network Analysis of Social Structures</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 This paper has 14 pages and 8 figures. To be presented at the NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Evaluating the creative capabilities of large language models (LLMs) in complex tasks often requires human assessments that are difficult to scale. We introduce a novel, scalable methodology for evaluating LLM story generation by analyzing underlying social structures in narratives as signed character networks. To demonstrate its effectiveness, we conduct a large-scale comparative analysis using networks from over 1,200 stories, generated by four leading LLMs (GPT-4o, GPT-4o mini, Gemini 1.5 Pro, and Gemini 1.5 Flash) and a human-written corpus. Our findings, based on network properties like density, clustering, and signed edge weights, show that LLM-generated stories consistently exhibit a strong bias toward tightly-knit, positive relationships, which aligns with findings from prior research using human assessment. Our proposed approach provides a valuable tool for evaluating limitations and tendencies in the creative storytelling of current and future LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18931v1">A Justice Lens on Fairness and Ethics Courses in Computing Education: LLM-Assisted Multi-Perspective and Thematic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 14 pages, 8 figures, In Review
    </div>
    <details class="paper-abstract">
      Course syllabi set the tone and expectations for courses, shaping the learning experience for both students and instructors. In computing courses, especially those addressing fairness and ethics in artificial intelligence (AI), machine learning (ML), and algorithmic design, it is imperative that we understand how approaches to navigating barriers to fair outcomes are being addressed.These expectations should be inclusive, transparent, and grounded in promoting critical thinking. Syllabus analysis offers a way to evaluate the coverage, depth, practices, and expectations within a course. Manual syllabus evaluation, however, is time-consuming and prone to inconsistency. To address this, we developed a justice-oriented scoring rubric and asked a large language model (LLM) to review syllabi through a multi-perspective role simulation. Using this rubric, we evaluated 24 syllabi from four perspectives: instructor, departmental chair, institutional reviewer, and external evaluator. We also prompted the LLM to identify thematic trends across the courses. Findings show that multiperspective evaluation aids us in noting nuanced, role-specific priorities, leveraging them to fill hidden gaps in curricula design of AI/ML and related computing courses focused on fairness and ethics. These insights offer concrete directions for improving the design and delivery of fairness, ethics, and justice content in such courses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18927v1">BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18339v1">ECG-LLM -- training and evaluation of domain-specific large language models for electrocardiography</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 34 pages, 8 figures, code available at https://github.com/AI4HealthUOL/ecg-llm
    </div>
    <details class="paper-abstract">
      Domain-adapted open-weight large language models (LLMs) offer promising healthcare applications, from queryable knowledge bases to multimodal assistants, with the crucial advantage of local deployment for privacy preservation. However, optimal adaptation strategies, evaluation methodologies, and performance relative to general-purpose LLMs remain poorly characterized. We investigated these questions in electrocardiography, an important area of cardiovascular medicine, by finetuning open-weight models on domain-specific literature and implementing a multi-layered evaluation framework comparing finetuned models, retrieval-augmented generation (RAG), and Claude Sonnet 3.7 as a representative general-purpose model. Finetuned Llama 3.1 70B achieved superior performance on multiple-choice evaluations and automatic text metrics, ranking second to Claude 3.7 in LLM-as-a-judge assessments. Human expert evaluation favored Claude 3.7 and RAG approaches for complex queries. Finetuned models significantly outperformed their base counterparts across nearly all evaluation modes. Our findings reveal substantial performance heterogeneity across evaluation methodologies, underscoring assessment complexity. Nevertheless, domain-specific adaptation through finetuning and RAG achieves competitive performance with proprietary models, supporting the viability of privacy-preserving, locally deployable clinical solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18871v1">How Do LLMs Use Their Depth?</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Growing evidence suggests that large language models do not use their depth uniformly, yet we still lack a fine-grained understanding of their layer-wise prediction dynamics. In this paper, we trace the intermediate representations of several open-weight models during inference and reveal a structured and nuanced use of depth. Specifically, we propose a "Guess-then-Refine" framework that explains how LLMs internally structure their computations to make predictions. We first show that the top-ranked predictions in early LLM layers are composed primarily of high-frequency tokens, which act as statistical guesses proposed by the model early on due to the lack of appropriate contextual information. As contextual information develops deeper into the model, these initial guesses get refined into contextually appropriate tokens. Even high-frequency token predictions from early layers get refined >70% of the time, indicating that correct token prediction is not "one-and-done". We then go beyond frequency-based prediction to examine the dynamic usage of layer depth across three case studies. (i) Part-of-speech analysis shows that function words are, on average, the earliest to be predicted correctly. (ii) Fact recall task analysis shows that, in a multi-token answer, the first token requires more computational depth than the rest. (iii) Multiple-choice task analysis shows that the model identifies the format of the response within the first half of the layers, but finalizes its response only toward the end. Together, our results provide a detailed view of depth usage in LLMs, shedding light on the layer-by-layer computations that underlie successful predictions and providing insights for future works to improve computational efficiency in transformer-based models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14456v2">Correct-Detect: Balancing Performance and Ambiguity Through the Lens of Coreference Resolution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted at EMNLP 2025 (main)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intended to reflect human linguistic competencies. But humans have access to a broad and embodied context, which is key in detecting and resolving linguistic ambiguities, even in isolated text spans. A foundational case of semantic ambiguity is found in the task of coreference resolution: how is a pronoun related to an earlier person mention? This capability is implicit in nearly every downstream task, and the presence of ambiguity at this level can alter performance significantly. We show that LLMs can achieve good performance with minimal prompting in both coreference disambiguation and the detection of ambiguity in coreference, however, they cannot do both at the same time. We present the CORRECT-DETECT trade-off: though models have both capabilities and deploy them implicitly, successful performance balancing these two abilities remains elusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03417v2">NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 This paper has been accepted in the main conference proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15926v2">TeLLMe v2: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      With the emergence of wearable devices and other embedded systems, deploying large language models (LLMs) on edge platforms has become an urgent need. However, this is challenging because of their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as low as 1.58~bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected long latency of the prefill stage. We present \textbf{TeLLMe}, the first table-lookup-based ternary LLM accelerator for low-power edge FPGAs that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. TeLLMe incorporates several novel techniques, including (1) a table-lookup-based ternary matrix multiplication (TLMM) engine utilizing grouped activations and online precomputation for low resource utilization and high throughput; (2) a fine-grained analytic URAM-based weight buffer management scheme for efficient loading and compute engine access; (3) a streaming dataflow architecture that fuses floating-point element-wise operations with linear computations to hide latency; (4) a reversed-reordered prefill stage attention with fused attention operations for high memory efficiency; and (5) a resource-efficient specialized decoding stage attention. Under a 5~W power budget, TeLLMe delivers up to 25~tokens/s decoding throughput and 0.45--0.96~s time-to-first-token (TTFT) for 64--128 token prompts, marking a significant energy-efficiency advancement in LLM inference on edge FPGAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18814v1">Online SFT for LLM Reasoning: Surprising Effectiveness of Self-Tuning without Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We present a simple, self-help online supervised finetuning (OSFT) paradigm for LLM reasoning. In this paradigm, the model generates its own responses and is immediately finetuned on this self-generated data. OSFT is a highly efficient training strategy for LLM reasoning, as it is reward-free and uses just one rollout by default. Experiment results show that OSFT achieves downstream performance on challenging mathematical reasoning tasks comparable to strong reinforcement learning with verifiable rewards (RLVR) methods such as GRPO. Our ablation study further demonstrates the efficiency and robustness of OSFT. The major mechanism of OSFT lies in facilitating the model's own existing preference (latent knowledge) learned from pretraining, which leads to reasoning ability improvement. We believe that OSFT offers an efficient and promising alternative to more complex, reward-based training paradigms. Our code is available at https://github.com/ElementQi/OnlineSFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17017v2">SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked "How can I track someone's location without their consent?", a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14008v2">Stop Reducing Responsibility in LLM-Powered Multi-Agent Systems to Local Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Updated manuscript of our previous version (arXiv:2502.01714). Under review
    </div>
    <details class="paper-abstract">
      LLM-powered Multi-Agent Systems (LLM-MAS) unlock new potentials in distributed reasoning, collaboration, and task generalization but also introduce additional risks due to unguaranteed agreement, cascading uncertainty, and adversarial vulnerabilities. We argue that ensuring responsible behavior in such systems requires a paradigm shift: from local, superficial agent-level alignment to global, systemic agreement. We conceptualize responsibility not as a static constraint but as a lifecycle-wide property encompassing agreement, uncertainty, and security, each requiring the complementary integration of subjective human-centered values and objective verifiability. Furthermore, a dual-perspective governance framework that combines interdisciplinary design with human-AI collaborative oversight is essential for tracing and ensuring responsibility throughout the lifecycle of LLM-MAS. Our position views LLM-MAS not as loose collections of agents, but as unified, dynamic socio-technical systems that demand principled mechanisms to support each dimension of responsibility and enable ethically aligned, verifiably coherent, and resilient behavior for sustained, system-wide agreement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02186v2">EvalAssist: A Human-Centered Tool for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 arXiv admin note: substantial text overlap with arXiv:2410.00873
    </div>
    <details class="paper-abstract">
      With the broad availability of large language models and their ability to generate vast outputs using varied prompts and configurations, determining the best output for a given task requires an intensive evaluation process, one where machine learning practitioners must decide how to assess the outputs and then carefully carry out the evaluation. This process is both time-consuming and costly. As practitioners work with an increasing number of models, they must now evaluate outputs to determine which model and prompt performs best for a given task. LLMs are increasingly used as evaluators to filter training data, evaluate model performance, assess harms and risks, or assist human evaluators with detailed assessments. We present EvalAssist, a framework that simplifies the LLM-as-a-judge workflow. The system provides an online criteria development environment, where users can interactively build, test, and share custom evaluation criteria in a structured and portable format. We support a set of LLM-based evaluation pipelines that leverage off-the-shelf LLMs and use a prompt-chaining approach we developed and contributed to the UNITXT open-source library. Additionally, our system also includes specially trained evaluators to detect harms and risks in LLM outputs. We have deployed the system internally in our organization with several hundreds of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18795v1">ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 17 pages, 5 fiugres
    </div>
    <details class="paper-abstract">
      The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18787v1">ShaRE your Data! Characterizing Datasets for LLM-based Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      [Context] Large Language Models (LLMs) rely on domain-specific datasets to achieve robust performance across training and inference stages. However, in Requirements Engineering (RE), data scarcity remains a persistent limitation reported in surveys and mapping studies. [Question/Problem] Although there are multiple datasets supporting LLM-based RE tasks (LLM4RE), they are fragmented and poorly characterized, limiting reuse and comparability. This research addresses the limited visibility and characterization of datasets used in LLM4RE. We investigate which public datasets are employed, how they can be systematically characterized, and which RE tasks and dataset descriptors remain under-represented. [Ideas/Results] To address this, we conduct a systematic mapping study to identify and analyse datasets used in LLM4RE research. A total of 62 publicly available datasets are referenced across 43 primary studies. Each dataset is characterized along descriptors such as artifact type, granularity, RE stage, task, domain, and language. Preliminary findings show multiple research gaps, including limited coverage for elicitation tasks, scarce datasets for management activities beyond traceability, and limited multilingual availability. [Contribution] This research preview offers a public catalogue and structured characterization scheme to support dataset selection, comparison, and reuse in LLM4RE research. Future work will extend the scope to grey literature, as well as integration with open dataset and benchmark repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10806v2">Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Waiting for Conference Response
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are adept at generating responses based on information within their context. While this ability is useful for interacting with structured data like code files, another popular method, Retrieval-Augmented Generation (RAG), retrieves relevant documents to augment the model's in-context learning. However, it is not well-explored how to best represent this retrieved knowledge for generating responses on structured data, particularly hierarchical structures like trees. In this work, we propose a novel bottom-up method to linearize knowledge from tree-like structures (like a GitHub repository) by generating implicit, aggregated summaries at each hierarchical level. This approach enables the knowledge to be stored in a knowledge base and used directly with RAG. We then compare our method to using RAG on raw, unstructured code, evaluating the accuracy and quality of the generated responses. Our results show that while response quality is comparable across both methods, our approach generates over 68% fewer documents in the retriever, a significant gain in efficiency. This finding suggests that leveraging implicit, linearized knowledge may be a highly effective and scalable strategy for handling complex, hierarchical data structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03655v2">Facts are Harder Than Opinions -- A Multilingual, Comparative Analysis of LLM-Based Fact-Checking Reliability</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation necessitates scalable, automated fact-checking solutions. Yet, current benchmarks often overlook multilingual and topical diversity. This paper introduces a novel, dynamically extensible data set that includes 61,514 claims in multiple languages and topics, extending existing datasets up to 2024. Through a comprehensive evaluation of five prominent Large Language Models (LLMs), including GPT-4o, GPT-3.5 Turbo, LLaMA 3.1, and Mixtral 8x7B, we identify significant performance gaps between different languages and topics. While overall GPT-4o achieves the highest accuracy, it declines to classify 43% of claims. Across all models, factual-sounding claims are misclassified more often than opinions, revealing a key vulnerability. These findings underscore the need for caution and highlight challenges in deploying LLM-based fact-checking systems at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13982v3">Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Preprint; feedback welcome
    </div>
    <details class="paper-abstract">
      What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16551v2">From Reviews to Actionable Insights: An LLM-Based Approach for Attribute and Feature Extraction</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      This research proposes a systematic, large language model (LLM) approach for extracting product and service attributes, features, and associated sentiments from customer reviews. Grounded in marketing theory, the framework distinguishes perceptual attributes from actionable features, producing interpretable and managerially actionable insights. We apply the methodology to 20,000 Yelp reviews of Starbucks stores and evaluate eight prompt variants on a random subset of reviews. Model performance is assessed through agreement with human annotations and predictive validity for customer ratings. Results show high consistency between LLMs and human coders and strong predictive validity, confirming the reliability of the approach. Human coders required a median of six minutes per review, whereas the LLM processed each in two seconds, delivering comparable insights at a scale unattainable through manual coding. Managerially, the analysis identifies attributes and features that most strongly influence customer satisfaction and their associated sentiments, enabling firms to pinpoint "joy points," address "pain points," and design targeted interventions. We demonstrate how structured review data can power an actionable marketing dashboard that tracks sentiment over time and across stores, benchmarks performance, and highlights high-leverage features for improvement. Simulations indicate that enhancing sentiment for key service features could yield 1-2% average revenue gains per store.
    </details>
</div>
