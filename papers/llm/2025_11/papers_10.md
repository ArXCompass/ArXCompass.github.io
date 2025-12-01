# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- Part 10
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03980v1">LLMs and Cultural Values: the Impact of Prompt Language and Explicit Cultural Framing</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Preprint under review at Computational Linguistics. Accepted with minor revisions (10/10/2025); second round
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly being adopted by users across the globe, who interact with them in a diverse range of languages. At the same time, there are well-documented imbalances in the training data and optimisation objectives of this technology, raising doubts as to whether LLMs can represent the cultural diversity of their broad user base. In this study, we look at LLMs and cultural values and examine how prompt language and cultural framing influence model responses and their alignment with human values in different countries. We probe 10 LLMs with 63 items from the Hofstede Values Survey Module and World Values Survey, translated into 11 languages, and formulated as prompts with and without different explicit cultural perspectives. Our study confirms that both prompt language and cultural perspective produce variation in LLM outputs, but with an important caveat: While targeted prompting can, to a certain extent, steer LLM responses in the direction of the predominant values of the corresponding countries, it does not overcome the models' systematic bias toward the values associated with a restricted set of countries in our dataset: the Netherlands, Germany, the US, and Japan. All tested models, regardless of their origin, exhibit remarkably similar patterns: They produce fairly neutral responses on most topics, with selective progressive stances on issues such as social tolerance. Alignment with cultural values of human respondents is improved more with an explicit cultural perspective than with a targeted prompt language. Unexpectedly, combining both approaches is no more effective than cultural framing with an English prompt. These findings reveal that LLMs occupy an uncomfortable middle ground: They are responsive enough to changes in prompts to produce variation, but too firmly anchored to specific cultural defaults to adequately represent cultural diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03934v1">PEFA-AI: Advancing Open-source LLMs for RTL generation using Progressive Error Feedback Agentic-AI</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Appeared in the Design Automation Conference (DAC) 2025, Workshop Poster on June 22, 2025
    </div>
    <details class="paper-abstract">
      We present an agentic flow consisting of multiple agents that combine specialized LLMs and hardware simulation tools to collaboratively complete the complex task of Register Transfer Level (RTL) generation without human intervention. A key feature of the proposed flow is the progressive error feedback system of agents (PEFA), a self-correcting mechanism that leverages iterative error feedback to progressively increase the complexity of the approach. The generated RTL includes checks for compilation, functional correctness, and synthesizable constructs. To validate this adaptive approach to code generation, benchmarking is performed using two opensource natural language-to-RTL datasets. We demonstrate the benefits of the proposed approach implemented on an open source agentic framework, using both open- and closed-source LLMs, effectively bridging the performance gap between them. Compared to previously published methods, our approach sets a new benchmark, providing state-of-the-art pass rates while being efficient in token counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22913v2">Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 20 pages, 9 figures, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We demonstrate that unstructured sparsity significantly improves KV cache compression for LLMs, enabling sparsity levels up to 70% without compromising accuracy or requiring fine-tuning. We conduct a systematic exploration of pruning strategies and find per-token magnitude-based pruning as highly effective for both Key and Value caches under unstructured sparsity, surpassing prior structured pruning schemes. The Key cache benefits from prominent outlier elements, while the Value cache surprisingly benefits from a simple magnitude-based pruning despite its uniform distribution. KV cache size is the major bottleneck in decode performance due to high memory overhead for large context lengths. To address this, we use a bitmap-based sparse format and a custom attention kernel capable of compressing and directly computing over compressed caches pruned to arbitrary sparsity patterns, significantly accelerating memory-bound operations in decode computations and thereby compensating for the overhead of runtime pruning and compression. Our custom attention kernel coupled with the bitmap-based format delivers substantial compression of KV cache upto 45% of dense inference and thereby enables longer context length and increased tokens/sec throughput of upto 2.23x compared to dense inference. Our pruning mechanism and sparse attention kernel is available at https://github.com/dhjoo98/mustafar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04875v1">Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04869v1">Trained on Tokens, Calibrated on Concepts: The Emergence of Semantic Calibration in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often lack meaningful confidence estimates for their outputs. While base LLMs are known to exhibit next-token calibration, it remains unclear whether they can assess confidence in the actual meaning of their responses beyond the token level. We find that, when using a certain sampling-based notion of semantic calibration, base LLMs are remarkably well-calibrated: they can meaningfully assess confidence in open-domain question-answering tasks, despite not being explicitly trained to do so. Our main theoretical contribution establishes a mechanism for why semantic calibration emerges as a byproduct of next-token prediction, leveraging a recent connection between calibration and local loss optimality. The theory relies on a general definition of "B-calibration," which is a notion of calibration parameterized by a choice of equivalence classes (semantic or otherwise). This theoretical mechanism leads to a testable prediction: base LLMs will be semantically calibrated when they can easily predict their own distribution over semantic answer classes before generating a response. We state three implications of this prediction, which we validate through experiments: (1) Base LLMs are semantically calibrated across question-answering tasks, (2) RL instruction-tuning systematically breaks this calibration, and (3) chain-of-thought reasoning breaks calibration. To our knowledge, our work provides the first principled explanation of when and why semantic calibration emerges in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15809v2">Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 AACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23842v3">Fair Document Valuation in LLM Summaries via Shapley Values</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these systems enhance user experience through coherent summaries, they obscure the individual contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries by proposing a Shapley value-based framework for fair document valuation. Although theoretically appealing, exact Shapley value computation is prohibitively expensive at scale. To improve efficiency, we develop Cluster Shapley, a simple approximation algorithm that leverages semantic similarity among documents to reduce computation while maintaining attribution accuracy. Using Amazon product review data, we empirically show that off-the-shelf Shapley approximations, such as Monte Carlo sampling and Kernel SHAP, perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency-accuracy frontier. Moreover, simple attribution rules (e.g., equal or relevance-based allocation), though computationally cheap, lead to highly unfair outcomes. Together, our findings highlight the potential of structure-aware Shapley approximations tailored to LLM summarization and offer guidance for platforms seeking scalable and fair content attribution mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04847v1">Grounded Test-Time Adaptation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions. This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time. To address these issues, we propose two distinct and complementary strategies for adapting LLM agents by leveraging environment-specific information available during deployment. First, an online distributional adaptation method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format. Second, a deployment-time dynamics grounding method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with a nonparametric world model. We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation. Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost. We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents. For example, on the WebArena multi-site split, this method increases the agent's success rate from 2% to 23%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16204v2">Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 To Appear in NeurIPS 2025 Datasets & Benchmarks Track
    </div>
    <details class="paper-abstract">
      Modern engineering, spanning electrical, mechanical, aerospace, civil, and computer disciplines, stands as a cornerstone of human civilization and the foundation of our society. However, engineering design poses a fundamentally different challenge for large language models (LLMs) compared with traditional textbook-style problem solving or factual question answering. Although existing benchmarks have driven progress in areas such as language understanding, code synthesis, and scientific problem solving, real-world engineering design demands the synthesis of domain knowledge, navigation of complex trade-offs, and management of the tedious processes that consume much of practicing engineers' time. Despite these shared challenges across engineering disciplines, no benchmark currently captures the unique demands of engineering design work. In this work, we introduce EngDesign, an Engineering Design benchmark that evaluates LLMs' abilities to perform practical design tasks across nine engineering domains. Unlike existing benchmarks that focus on factual recall or question answering, EngDesign uniquely emphasizes LLMs' ability to synthesize domain knowledge, reason under constraints, and generate functional, objective-oriented engineering designs. Each task in EngDesign represents a real-world engineering design problem, accompanied by a detailed task description specifying design goals, constraints, and performance requirements. EngDesign pioneers a simulation-based evaluation paradigm that moves beyond textbook knowledge to assess genuine engineering design capabilities and shifts evaluation from static answer checking to dynamic, simulation-driven functional verification, marking a crucial step toward realizing the vision of engineering Artificial General Intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24095v2">SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) efficiently in multi-region setups remains a challenge. Due to cost and GPU availability concerns, providers typically deploy LLMs in multiple regions using instance with long-term commitments, like reserved instances or on-premise clusters, which are often underutilized due to their region-local traffic handling and diurnal traffic variance. In this paper, we introduce SkyWalker, a multi-region load balancer for LLM inference that aggregates regional diurnal patterns through cross-region traffic handling. By doing so, SkyWalker enables providers to reserve instances based on expected global demand, rather than peak demand in each individual region. Meanwhile, SkyWalker preserves KV-Cache locality and load balancing, ensuring cost efficiency without sacrificing performance. SkyWalker achieves this with a cache-aware cross-region traffic handler and a selective pushing based load balancing mechanism. Our evaluation on real-world workloads shows that it achieves 1.12-2.06x higher throughput and 1.74-6.30x lower latency compared to existing load balancers, while reducing total serving cost by 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04791v1">DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Modern LLM serving systems must sustain high throughput while meeting strict latency SLOs across two distinct inference phases: compute-intensive prefill and memory-bound decode phases. Existing approaches either (1) aggregate both phases on shared GPUs, leading to interference between prefill and decode phases, which degrades time-between-tokens (TBT); or (2) disaggregate the two phases across GPUs, improving latency but wasting resources through duplicated models and KV cache transfers. We present DuetServe, a unified LLM serving framework that achieves disaggregation-level isolation within a single GPU. DuetServe operates in aggregated mode by default and dynamically activates SM-level GPU spatial multiplexing when TBT degradation is predicted. Its key idea is to decouple prefill and decode execution only when needed through fine-grained, adaptive SM partitioning that provides phase isolation only when contention threatens latency service level objectives (SLOs). DuetServe integrates (1) an attention-aware roofline model to forecast iteration latency, (2) a partitioning optimizer that selects the optimal SM split to maximize throughput under TBT constraints, and (3) an interruption-free execution engine that eliminates CPU-GPU synchronization overhead. Evaluations show that DuetServe improves total throughput by up to 1.3x while maintaining low generation latency compared to state-of-the-art frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05447v3">TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance Beyond RAG</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is one of the leading and most widely used techniques for enhancing LLM retrieval capabilities, but it still faces significant limitations in commercial use cases. RAG primarily relies on the query-chunk text-to-text similarity in the embedding space for retrieval and can fail to capture deeper semantic relationships across chunks, is highly sensitive to chunking strategies, and is prone to hallucinations. To address these challenges, we propose TOBUGraph, a graph-based retrieval framework that first constructs the knowledge graph from unstructured data dynamically and automatically. Using LLMs, TOBUGraph extracts structured knowledge and diverse relationships among data, going beyond RAG's text-to-text similarity. Retrieval is achieved through graph traversal, leveraging the extracted relationships and structures to enhance retrieval accuracy, eliminating the need for chunking configurations while reducing hallucination. We demonstrate TOBUGraph's effectiveness in TOBU, a real-world application in production for personal memory organization and retrieval. Our evaluation using real user data demonstrates that TOBUGraph outperforms multiple RAG implementations in both precision and recall, significantly improving user experience through improved retrieval accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.20749v3">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks. Our code is publicly available at: https://github.com/lichangh20/Matryoshka.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05410v2">Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 14 pages,7 figures;Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights ({\it local homogeneity}), adjacent values demonstrate distinct {\it heterogeneous} distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin.Our code can be found in this link:https://github.com/the-scale-lab/Asymkv.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04541v1">LLM-as-a-Judge: Toward World Models for Slate Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Modeling user preferences across domains remains a key challenge in slate recommendation (i.e. recommending an ordered sequence of items) research. We investigate how Large Language Models (LLM) can effectively act as world models of user preferences through pairwise reasoning over slates. We conduct an empirical study involving several LLMs on three tasks spanning different datasets. Our results reveal relationships between task performance and properties of the preference function captured by LLMs, hinting towards areas for improvement and highlighting the potential of LLMs as world models in recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04538v1">From Model to Breach: Towards Actionable LLM-Generated Vulnerabilities Reporting</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      As the role of Large Language Models (LLM)-based coding assistants in software development becomes more critical, so does the role of the bugs they generate in the overall cybersecurity landscape. While a number of LLM code security benchmarks have been proposed alongside approaches to improve the security of generated code, it remains unclear to what extent they have impacted widely used coding LLMs. Here, we show that even the latest open-weight models are vulnerable in the earliest reported vulnerability scenarios in a realistic use setting, suggesting that the safety-functionality trade-off has until now prevented effective patching of vulnerabilities. To help address this issue, we introduce a new severity metric that reflects the risk posed by an LLM-generated vulnerability, accounting for vulnerability severity, generation chance, and the formulation of the prompt that induces vulnerable code generation - Prompt Exposure (PE). To encourage the mitigation of the most serious and prevalent vulnerabilities, we use PE to define the Model Exposure (ME) score, which indicates the severity and prevalence of vulnerabilities a model generates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17737v2">LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Paper accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      While state-of-the-art large language models (LLMs) have shown impressive performance on many tasks, there has been extensive research on undesirable model behavior such as hallucinations and bias. In this work, we investigate how the quality of LLM responses changes in terms of information accuracy, truthfulness, and refusals depending on three user traits: English proficiency, education level, and country of origin. We present extensive experimentation on three state-of-the-art LLMs and two different datasets targeting truthfulness and factuality. Our findings suggest that undesirable behaviors in state-of-the-art LLMs occur disproportionately more for users with lower English proficiency, of lower education status, and originating from outside the US, rendering these models unreliable sources of information towards their most vulnerable users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04491v1">RUST-BENCH: Benchmarking LLM Reasoning on Unstructured Text within Structured Tables</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Existing tabular reasoning benchmarks mostly test models on small, uniform tables, underrepresenting the complexity of real-world data and giving an incomplete view of Large Language Models' (LLMs) reasoning abilities. Real tables are long, heterogeneous, and domain-specific, mixing structured fields with free text and requiring multi-hop reasoning across thousands of tokens. To address this gap, we introduce RUST-BENCH, a benchmark of 7966 questions from 2031 real-world tables spanning two domains: i) RB-Science (NSF grant records) and ii) RB-Sports (NBA statistics). Unlike prior work, RUST-BENCH evaluates LLMs jointly across scale, heterogeneity, domain specificity, and reasoning complexity. Experiments with open-source and proprietary models show that LLMs struggle with heterogeneous schemas and complex multi-hop inference, revealing persistent weaknesses in current architectures and prompting strategies. RUST-BENCH establishes a challenging new testbed for advancing tabular reasoning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04486v1">EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Instructed code editing, where LLMs directly modify a developer's existing code based on a user instruction, is becoming a widely used interaction mode in AI coding assistants. However, few benchmarks directly evaluate this capability and current datasets often rely on artificial sources. We introduce EDIT-Bench, a benchmark for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EDIT-Bench comprises of 545 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. EDIT-Bench introduces context-dependent problems that require the model to understand code context, highlighted code, and cursor position in addition to the user instruction. We evaluate 40 diverse LLMs and observe that EDIT-Bench is a challenging set of problems where only 5 models score over 60%. We find that model performance varies across different categories of user instructions. Further, we find that varying levels of contextual information greatly affect task success rate, with performance varying up to 11%, indicating the importance of evaluating with realistic context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04478v1">Generate, Evaluate, Iterate: Synthetic Data for Human-in-the-Loop Refinement of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 29 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm enables flexible, user-defined evaluation, but its effectiveness is often limited by the scarcity of diverse, representative data for refining criteria. We present a tool that integrates synthetic data generation into the LLM-as-a-judge workflow, empowering users to create tailored and challenging test cases with configurable domains, personas, lengths, and desired outcomes, including borderline cases. The tool also supports AI-assisted inline editing of existing test cases. To enhance transparency and interpretability, it reveals the prompts and explanations behind each generation. In a user study (N=24), 83% of participants preferred the tool over manually creating or selecting test cases, as it allowed them to rapidly generate diverse synthetic data without additional workload. The generated synthetic data proved as effective as hand-crafted data for both refining evaluation criteria and aligning with human preferences. These findings highlight synthetic data as a promising alternative, particularly in contexts where efficiency and scalability are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04477v1">Enabling Dynamic Sparsity in Quantized LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on end-user devices is gaining importance due to benefits in responsiveness, privacy, and operational cost. Yet the limited memory and compute capability of mobile and desktop GPUs make efficient execution difficult. Recent observations suggest that the internal activations of LLMs are often dynamically sparse, meaning that for each input, only part of the network contributes significantly to the output. Such sparsity could reduce computation, but it interacts poorly with group-wise quantization, which remains the dominant approach for fitting LLMs onto resource-constrained hardware. To reconcile these two properties, this study proposes a set of techniques that realize dynamic sparse inference under low-bit quantization. The method features: (1) a zigzag-patterned quantization layout that organizes weights in a way consistent with activation sparsity and improves GPU memory locality; (2) a specialized GEMV kernel designed for this layout to fully utilize parallel compute units; and (3) a compact runtime mechanism that gathers sparse indices with minimal overhead. Across several model scales and hardware configurations, the approach achieves up to 1.55x faster decoding throughput while maintaining accuracy comparable to dense quantized inference, showing that structured sparsity and quantization can effectively coexist on commodity GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04847v2">Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 EMNLP Industry Track 2025
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) aims to reduce hallucinations by grounding responses in external context, yet large language models (LLMs) still frequently introduce unsupported information or contradictions even when provided with relevant context. This paper presents two complementary efforts at Vectara to measure and benchmark LLM faithfulness in RAG. First, we describe our original hallucination leaderboard, which has tracked hallucination rates for LLMs since 2023 using our HHEM hallucination detection model. Motivated by limitations observed in current hallucination detection methods, we introduce FaithJudge, an LLM-as-a-judge framework that leverages a pool of diverse human-annotated hallucination examples to substantially improve the automated hallucination evaluation of LLMs. We introduce an enhanced hallucination leaderboard centered on FaithJudge that benchmarks LLMs on RAG faithfulness in summarization, question-answering, and data-to-text generation tasks. FaithJudge enables a more reliable benchmarking of LLM hallucinations in RAG and supports the development of more trustworthy generative AI systems: https://github.com/vectara/FaithJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04473v1">Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, a framework for generating high-quality synthetic Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over each question. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models. We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00783v2">When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 This paper has been submitted to IEEE Transactions on Mobile Computing. Jingzehua Xu, Weihang Zhang, and Yangyang Li contributed equally to this work and are recognized as the co-first authors of the paper
    </div>
    <details class="paper-abstract">
      Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06991v2">Evaluating LLM-Contaminated Crowdsourcing Data Without Ground Truth</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 32 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The recent success of generative AI highlights the crucial role of high-quality human feedback in building trustworthy AI systems. However, the increasing use of large language models (LLMs) by crowdsourcing workers poses a significant challenge: datasets intended to reflect human input may be compromised by LLM-generated responses. Existing LLM detection approaches often rely on high-dimensional training data such as text, making them unsuitable for annotation tasks like multiple-choice labeling. In this work, we investigate the potential of peer prediction -- a mechanism that evaluates the information within workers' responses without using ground truth -- to mitigate LLM-assisted cheating in crowdsourcing with a focus on annotation tasks. Our approach quantifies the correlations between worker answers while conditioning on (a subset of) LLM-generated labels available to the requester. Building on prior research, we propose a training-free scoring mechanism with theoretical guarantees under a crowdsourcing model that accounts for LLM collusion. We establish conditions under which our method is effective and empirically demonstrate its robustness in detecting low-effort cheating on real-world crowdsourcing datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04432v1">If I Could Turn Back Time: Temporal Reframing as a Historical Reasoning Task for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 8 pages, 1 figure, 3 tables, submitted to aconference
    </div>
    <details class="paper-abstract">
      In this study, we experiment with the ability of LLMs to do temporal reasoning. Using a Norwegian book from 1940 containing trivia questions, we prompt the LLMs to answer the questions as if it were 1940. We also pose the questions in both English and Norwegian. Correct answers are often presented as sentences, and grading is done by means of LLM-as-judge, with sampled checks by a native speaker. Prompting in English consistently gave better results than in Norwegian, an unexpected result. In contrast, using larger LLMs improved results. We tested the DeepSeek-R1, Gemma3, Qwen3, and Llama3.1 model families, and also the largest available LLM especially crafted for Norwegian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04427v1">Speed at the Cost of Quality? The Impact of LLM Agent Assistance on Software Development</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated the promise to revolutionize the field of software engineering. Among other things, LLM agents are rapidly gaining momentum in their application to software development, with practitioners claiming a multifold productivity increase after adoption. Yet, empirical evidence is lacking around these claims. In this paper, we estimate the causal effect of adopting a widely popular LLM agent assistant, namely Cursor, on development velocity and software quality. The estimation is enabled by a state-of-the-art difference-in-differences design comparing Cursor-adopting GitHub projects with a matched control group of similar GitHub projects that do not use Cursor. We find that the adoption of Cursor leads to a significant, large, but transient increase in project-level development velocity, along with a significant and persistent increase in static analysis warnings and code complexity. Further panel generalized method of moments estimation reveals that the increase in static analysis warnings and code complexity acts as a major factor causing long-term velocity slowdown. Our study carries implications for software engineering practitioners, LLM agent assistant designers, and researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04418v1">The Illusion of Certainty: Uncertainty quantification for LLMs fails under ambiguity</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification (UQ) in Large Language Models (LLMs) is critical for trustworthy deployment. While real-world language is inherently ambiguous, reflecting aleatoric uncertainty, existing UQ methods are typically benchmarked against tasks with no ambiguity. In this work, we demonstrate that while current uncertainty estimators perform well under the restrictive assumption of no ambiguity, they degrade to close-to-random performance on ambiguous data. To this end, we introduce MAQA* and AmbigQA*, the first ambiguous question-answering (QA) datasets equipped with ground-truth answer distributions estimated from factual co-occurrence. We find this performance deterioration to be consistent across different estimation paradigms: using the predictive distribution itself, internal representations throughout the model, and an ensemble of models. We show that this phenomenon can be theoretically explained, revealing that predictive-distribution and ensemble-based estimators are fundamentally limited under ambiguity. Overall, our study reveals a key shortcoming of current UQ methods for LLMs and motivates a rethinking of current modeling paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04393v1">Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as "agents" for decision-making (DM) in interactive and dynamic environments. Yet, since they were not originally designed for DM, recent studies show that LLMs can struggle even in basic online DM problems, failing to achieve low regret or an effective exploration-exploitation tradeoff. To address this, we introduce Iterative Regret-Minimization Fine-Tuning (Iterative RMFT), a post-training procedure that repeatedly distills low-regret decision trajectories back into the base model. At each iteration, the model rolls out multiple decision trajectories, selects the k-lowest regret ones, and fine-tunes itself on them. Unlike prior methods that (a) distill action sequences from known DM algorithms or (b) rely on manually crafted chain-of-thought templates, our approach leverages the regret metric to elicit the model's own DM ability and reasoning rationales. This reliance on model-generated reasoning avoids rigid output engineering and provides more flexible, natural-language training signals. Empirical results show that Iterative RMFT improves LLMs' DM performance across diverse models - from Transformers with numerical input/output, to open-weight LLMs, and advanced closed-weight models like GPT-4o mini. Its flexibility in output and reasoning formats enables generalization across tasks with varying horizons, action spaces, reward processes, and natural-language contexts. Finally, we provide theoretical insight showing that a single-layer Transformer under this paradigm can act as a no-regret learner in a simplified setting. Overall, Iterative RMFT offers a principled and general post-training framework for enhancing LLMs' decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01827v3">APRMCTS: Improving LLM-based Automated Program Repair with Iterative Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) attempts to fix software bugs without human intervention, which plays a crucial role in software development and maintenance. Recently, with the advances in Large Language Models (LLMs), a rapidly increasing number of APR techniques have been proposed with remarkable performance. However, existing LLM-based APR techniques typically adopt trial-and-error strategies, which suffer from two major drawbacks: (1) inherently limited patch effectiveness due to local exploration, and (2) low search efficiency due to redundant exploration. In this paper, we propose APRMCTS, which uses iterative tree search to improve LLM-based APR. APRMCTS incorporates Monte Carlo Tree Search (MCTS) into patch searching by performing a global evaluation of the explored patches and selecting the most promising one for subsequent refinement and generation. APRMCTS effectively resolves the problems of falling into local optima and thus helps improve the efficiency of patch searching. Our experiments on 835 bugs from Defects4J demonstrate that, when integrated with GPT-3.5, APRMCTS can fix a total of 201 bugs, which outperforms all state-of-the-art baselines. Besides, APRMCTS helps GPT-4o-mini, GPT-3.5, Yi-Coder-9B, and Qwen2.5-Coder-7B to fix 30, 27, 37, and 28 more bugs, respectively. More importantly, APRMCTS boasts a significant performance advantage while employing small patch size (16 and 32), notably fewer than the 500 and 10,000 patches adopted in previous studies. In terms of cost, compared to existing state-of-the-art LLM-based APR methods, APRMCTS has time and monetary costs of less than 20% and 50%, respectively. Our extensive study demonstrates that APRMCTS exhibits good effectiveness and efficiency, with particular advantages in addressing complex bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04355v1">Where Do LLMs Still Struggle? An In-Depth Analysis of Code Generation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 To be published in Proceedings of 2025 2nd IEEE/ACM International Conference on AI-powered Software (AIware), Data & Benchmark Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in code generation, and the race to improve their performance has become a central focus of AI research. Benchmarks and leaderboards are increasingly popular, offering quantitative rankings of LLMs. However, they provide limited insight into the tasks that LLMs consistently fail to solve - information that is crucial for understanding current limitations and guiding the development of more capable models. To address this gap, we examined code generation tasks across four popular benchmarks, identifying those that major LLMs are most likely to fail. To understand the causes of these failures, we investigated whether the static complexity of solution code contributes to them, followed by a systematic inspection of 114 tasks that LLMs consistently struggled with. Our analysis revealed four recurring patterns of weaknesses in LLMs, as well as common complications within benchmark tasks that most often lead to failure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04317v1">RISE-T2V: Rephrasing and Injecting Semantics with LLM for Expansive Text-to-Video Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 17 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Most text-to-video(T2V) diffusion models depend on pre-trained text encoders for semantic alignment, yet they often fail to maintain video quality when provided with concise prompts rather than well-designed ones. The primary issue lies in their limited textual semantics understanding. Moreover, these text encoders cannot rephrase prompts online to better align with user intentions, which limits both the scalability and usability of the models, To address these challenges, we introduce RISE-T2V, which uniquely integrates the processes of prompt rephrasing and semantic feature extraction into a single and seamless step instead of two separate steps. RISE-T2V is universal and can be applied to various pre-trained LLMs and video diffusion models(VDMs), significantly enhancing their capabilities for T2V tasks. We propose an innovative module called the Rephrasing Adapter, enabling diffusion models to utilize text hidden states during the next token prediction of the LLM as a condition for video generation. By employing a Rephrasing Adapter, the video generation model can implicitly rephrase basic prompts into more comprehensive representations that better match the user's intent. Furthermore, we leverage the powerful capabilities of LLMs to enable video generation models to accomplish a broader range of T2V tasks. Extensive experiments demonstrate that RISE-T2V is a versatile framework applicable to different video diffusion model architectures, significantly enhancing the ability of T2V models to generate high-quality videos that align with user intent. Visual results are available on the webpage at https://rise-t2v.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04316v1">AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14133v3">GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Accepted to NeurIPS 2025. Project page and demos: https://air-ml.org/project/gasp/
    </div>
    <details class="paper-abstract">
      LLMs have shown impressive capabilities across various natural language processing tasks, yet remain vulnerable to input prompts, known as jailbreak attacks, carefully designed to bypass safety guardrails and elicit harmful responses. Traditional methods rely on manual heuristics but suffer from limited generalizability. Despite being automatic, optimization-based attacks often produce unnatural prompts that can be easily detected by safety filters or require high computational costs due to discrete token optimization. In this paper, we introduce Generative Adversarial Suffix Prompter (GASP), a novel automated framework that can efficiently generate human-readable jailbreak prompts in a fully black-box setting. In particular, GASP leverages latent Bayesian optimization to craft adversarial suffixes by efficiently exploring continuous latent embedding spaces, gradually optimizing the suffix prompter to improve attack efficacy while balancing prompt coherence via a targeted iterative refinement procedure. Through comprehensive experiments, we show that GASP can produce natural adversarial prompts, significantly improving jailbreak success over baselines, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15835v3">Pragmatic Reasoning improves LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive potential in translating natural language (NL) instructions into program code. However, user instructions often contain inherent ambiguities, making it challenging for LLMs to generate code that accurately reflects the user's true intent. To address this challenge, researchers have proposed approaches that produce multiple candidates of the program code and then rerank them to identify the best solution. In this paper, we propose CodeRSA, a novel code candidate reranking mechanism built upon the Rational Speech Act (RSA) framework, designed to guide LLMs toward more comprehensive pragmatic reasoning about user intent. We evaluate CodeRSA using Llama-3-8B-Instruct and Qwen-2.5-7B-Instruct on two widely used code generation benchmarks, HumanEval and MBPP. Our experiment results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. These findings underscore the effectiveness of integrating pragmatic reasoning into code candidate reranking, offering a promising direction for enhancing code generation quality in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09334v2">ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 MLSys 2025
    </div>
    <details class="paper-abstract">
      Recent developments in large language models (LLMs) have demonstrated their remarkable proficiency in a range of tasks. Compared to in-house homogeneous GPU clusters, deploying LLMs in cloud environments with diverse types of GPUs is crucial for addressing the GPU shortage problem and being more cost-effective. However, the diversity of network environments and various GPU types on the cloud bring difficulties to achieving high-performance serving. In this work, we propose ThunderServe, a high-performance and cost-efficient LLM serving system for heterogeneous cloud environments. We introduce a novel scheduling algorithm, which optimizes the deployment plan of LLM serving to accommodate the heterogeneous resource and network bandwidth conditions in cloud environments. Furthermore, we propose a lightweight re-scheduling mechanism, designed to adapt to fluctuating online conditions (e.g., node failures, workload shifts) without the need for costly restarts of ongoing services. Empirical results in both heterogeneous cloud and homogeneous in-house environments reveal that ThunderServe delivers up to a 2.1$\times$ and on average a $1.7\times$ increase in throughput and achieves up to a 2.5$\times$ and on average a $1.5\times$ reduction in latency deadlines compared with state-of-the-art systems given the same price budget, suggesting opting for cloud services provides a more cost-efficient solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03080v1">PolyNorm: Few-Shot LLM-Based Text Normalization for Text-to-Speech</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 9 pages including appendix. EMNLP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Text Normalization (TN) is a key preprocessing step in Text-to-Speech (TTS) systems, converting written forms into their canonical spoken equivalents. Traditional TN systems can exhibit high accuracy, but involve substantial engineering effort, are difficult to scale, and pose challenges to language coverage, particularly in low-resource settings. We propose PolyNorm, a prompt-based approach to TN using Large Language Models (LLMs), aiming to reduce the reliance on manually crafted rules and enable broader linguistic applicability with minimal human intervention. Additionally, we present a language-agnostic pipeline for automatic data curation and evaluation, designed to facilitate scalable experimentation across diverse languages. Experiments across eight languages show consistent reductions in the word error rate (WER) compared to a production-grade-based system. To support further research, we release PolyNorm-Benchmark, a multilingual data set covering a diverse range of text normalization phenomena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05830v2">Finetuning LLMs for Human Behavior Prediction in Social Science Experiments</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer a powerful opportunity to simulate the results of social science experiments. In this work, we demonstrate that finetuning LLMs directly on individual-level responses from past experiments meaningfully improves the accuracy of such simulations across diverse social science domains. We construct SocSci210 via an automatic pipeline, a dataset comprising 2.9 million responses from 400,491 participants in 210 open-source social science experiments. Through finetuning, we achieve multiple levels of generalization. In completely unseen studies, our strongest model, Socrates-Qwen-14B, produces predictions that are 26% more aligned with distributions of human responses to diverse outcome questions under varying conditions relative to its base model (Qwen2.5-14B), outperforming GPT-4o by 13%. By finetuning on a subset of conditions in a study, generalization to new unseen conditions is particularly robust, improving by 71%. Since SocSci210 contains rich demographic information, we reduce demographic parity difference, a measure of bias, by 10.6% through finetuning. Because social sciences routinely generate rich, topic-specific datasets, our findings indicate that finetuning on such data could enable more accurate simulations for experimental hypothesis screening. We release our data, models and finetuning code at stanfordhci.github.io/socrates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13406v2">Collaboration Dynamics and Reliability Challenges of Multi-Agent LLM Systems in Finite Element Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based multi-agent systems are increasingly applied to automate computational workflows in science and engineering. However, how inter-agent dynamics influence reasoning quality and verification reliability remains unclear. We study these mechanisms using an AutoGen-based multi-agent framework for linear-elastic Finite Element Analysis (FEA), evaluating seven role configurations across four tasks under a fixed 12-turn conversation limit. From 1,120 controlled trials, we find that collaboration effectiveness depends more on functional complementarity than team size: the three-agent Coder-Executor-Critic configuration uniquely produced physically and visually correct solutions, while adding redundant reviewers reduced success rates. Yet three systematic failure modes persist: (1) affirmation bias, where the Rebuttal agent endorsed rather than challenged outputs (85-92% agreement, including errors); (2) premature consensus caused by redundant reviewers; and (3) a verification-validation gap where executable but physically incorrect code passed undetected. No agent combination successfully validated constitutive relations in complex tasks. Building on theories of functional diversity, role differentiation, and computational validation, we propose actionable design principles: (i) assign complementary agent roles, (ii) enforce multi-level validation (execution, specification, physics), and (iii) prevent early consensus through adversarial or trigger-based interaction control. These findings establish a principled foundation for designing trustworthy LLM collaborations in engineering workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03878v1">KnowThyself: An Agentic Assistant for LLM Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 5 pages, 1 figure, Accepted for publication at the Demonstration Track of the 40th AAAI Conference on Artificial Intelligence (AAAI 26)
    </div>
    <details class="paper-abstract">
      We develop KnowThyself, an agentic assistant that advances large language model (LLM) interpretability. Existing tools provide useful insights but remain fragmented and code-intensive. KnowThyself consolidates these capabilities into a chat-based interface, where users can upload models, pose natural language questions, and obtain interactive visualizations with guided explanations. At its core, an orchestrator LLM first reformulates user queries, an agent router further directs them to specialized modules, and the outputs are finally contextualized into coherent explanations. This design lowers technical barriers and provides an extensible platform for LLM inspection. By embedding the whole process into a conversational workflow, KnowThyself offers a robust foundation for accessible LLM interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15706v2">Communication Efficient LLM Pre-training with SparseLoCo</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 20 pages, 14 tables, 2 figures
    </div>
    <details class="paper-abstract">
      Communication-efficient distributed training algorithms have received considerable interest recently due to their benefits for training Large Language Models (LLMs) in bandwidth-constrained settings, such as across datacenters and over the internet. Despite reducing communication frequency, these methods still typically require communicating a full copy of the model's gradients-resulting in a communication bottleneck even for cross-datacenter links. Furthermore, they can slightly degrade performance compared to a naive AdamW DDP baseline. While quantization is often applied to reduce the pseudo-gradient's size, in the context of LLM pre-training, existing approaches have been unable to additionally leverage sparsification and have obtained limited quantization. In this work, we introduce SparseLoCo, a communication-efficient training algorithm for LLMs that effectively leverages error feedback with Top-k sparsification and 2-bit quantization to reach extreme sparsity as low as 1-3% while outperforming full-precision DiLoCo. Our key observations are that outer momentum can be locally approximated by an error feedback accumulator combined with aggressive sparsity, and that sparse aggregation can actually improve model performance. We empirically demonstrate in a range of communication-constrained LLM training settings that SparseLoCo provides significant benefits in both performance and communication cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03844v1">ASAP: an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 This work has been accepted to Workshop on ML for Systems at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Optimizing large-language model (LLM) training on distributed domain-specific accelerator systems presents significant challenges due to its complex optimization space. Existing optimization methods, however, rely on time-consuming manual tuning or resource-intensive black-box searches, which struggle to keep pace with the rapidly evolving LLM domain, leading to slow development and underutilized resources. To address this, we introduce ASAP, an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training. It is a multi-agent system, featuring Coordinator, Analyzer, and Proposal agents, which integrates LLM reasoning with insights from performance profiling tools, roofline analysis, and a knowledge base of best practices and successful past optimizations from human experts. Our proposed design can automate the diagnosis of performance bottlenecks and recommend optimized sharding configurations with reasoning, thus effectively improving the efficiency of distributed LLM training. Experiments have shown that the ASAP-generated sharding configurations can contribute up to 28% training step time reduction and 1.43 times throughput improvement. When combined with additional optimization from human experts, throughput can be further increased to 2.58 times. The proposed ASAP promises to provide a scalable and explainable methodology for AI-assisted performance engineering in large-scale LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03830v1">Divide, Cache, Conquer: Dichotomic Prompting for Efficient Multi-Label LLM-Based Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 9 pages, 8 figures
    </div>
    <details class="paper-abstract">
      We introduce a method for efficient multi-label text classification with large language models (LLMs), built on reformulating classification tasks as sequences of dichotomic (yes/no) decisions. Instead of generating all labels in a single structured response, each target dimension is queried independently, which, combined with a prefix caching mechanism, yields substantial efficiency gains for short-text inference without loss of accuracy. To demonstrate the approach, we focus on affective text analysis, covering 24 dimensions including emotions and sentiment. Using LLM-to-SLM distillation, a powerful annotator model (DeepSeek-V3) provides multiple annotations per text, which are aggregated to fine-tune smaller models (HerBERT-Large, CLARIN-1B, PLLuM-8B, Gemma3-1B). The fine-tuned models show significant improvements over zero-shot baselines, particularly on the dimensions seen during training. Our findings suggest that decomposing multi-label classification into dichotomic queries, combined with distillation and cache-aware inference, offers a scalable and effective framework for LLM-based classification. While we validate the method on affective states, the approach is general and applicable across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03825v1">How Different Tokenization Algorithms Impact LLMs and Transformer Models for Binary Code Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Publication Notice. This paper was published in the BAR 2025 Workshop (with NDSS 2025) and is for research and educational use. Copyright \c{opyright} 2025 Internet Society. All rights reserved. Personal/classroom reproduction is permitted with this notice and full paper citation. All other uses, including commercial, require prior written permission from the Internet Society
    </div>
    <details class="paper-abstract">
      Tokenization is fundamental in assembly code analysis, impacting intrinsic characteristics like vocabulary size, semantic coverage, and extrinsic performance in downstream tasks. Despite its significance, tokenization in the context of assembly code remains an underexplored area. This study aims to address this gap by evaluating the intrinsic properties of Natural Language Processing (NLP) tokenization models and parameter choices, such as vocabulary size. We explore preprocessing customization options and pre-tokenization rules tailored to the unique characteristics of assembly code. Additionally, we assess their impact on downstream tasks like function signature prediction -- a critical problem in binary code analysis. To this end, we conduct a thorough study on various tokenization models, systematically analyzing their efficiency in encoding assembly instructions and capturing semantic nuances. Through intrinsic evaluations, we compare tokenizers based on tokenization efficiency, vocabulary compression, and representational fidelity for assembly code. Using state-of-the-art pre-trained models such as the decoder-only Large Language Model (LLM) Llama 3.2, the encoder-only transformer BERT, and the encoder-decoder model BART, we evaluate the effectiveness of these tokenizers across multiple performance metrics. Preliminary findings indicate that tokenizer choice significantly influences downstream performance, with intrinsic metrics providing partial but incomplete predictability of extrinsic evaluation outcomes. These results reveal complex trade-offs between intrinsic tokenizer properties and their utility in practical assembly code tasks. Ultimately, this study provides valuable insights into optimizing tokenization models for low-level code analysis, contributing to the robustness and scalability of Natural Language Model (NLM)-based binary analysis workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03782v1">Expert Evaluation of LLM World Models: A High-$T_c$ Superconductivity Case Study</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 (v1) 9 pages, 4 figures, with 7-page supporting information. Accepted at the ICML 2025 workshop on Assessing World Models and the Explorations in AI Today workshop at ICML'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show great promise as a powerful tool for scientific literature exploration. However, their effectiveness in providing scientifically accurate and comprehensive answers to complex questions within specialized domains remains an active area of research. Using the field of high-temperature cuprates as an exemplar, we evaluate the ability of LLM systems to understand the literature at the level of an expert. We construct an expert-curated database of 1,726 scientific papers that covers the history of the field, and a set of 67 expert-formulated questions that probe deep understanding of the literature. We then evaluate six different LLM-based systems for answering these questions, including both commercially available closed models and a custom retrieval-augmented generation (RAG) system capable of retrieving images alongside text. Experts then evaluate the answers of these systems against a rubric that assesses balanced perspectives, factual comprehensiveness, succinctness, and evidentiary support. Among the six systems two using RAG on curated literature outperformed existing closed models across key metrics, particularly in providing comprehensive and well-supported answers. We discuss promising aspects of LLM performances as well as critical short-comings of all the models. The set of expert-formulated questions and the rubric will be valuable for assessing expert level performance of LLM based reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03758v1">Leveraging LLM-based agents for social science research: insights from citation network simulations</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 accepted by HSSCOMMS'25
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03706v1">LLM-enhanced Air Quality Monitoring Interface via Model Context Protocol</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 International Symposium on Advanced Electrical and Communication Technologies, ISAECT 2025
    </div>
    <details class="paper-abstract">
      Air quality monitoring is central to environmental sustainability and public health, yet traditional systems remain difficult for non-expert users to interpret due to complex visualizations, limited interactivity, and high deployment costs. Recent advances in Large Language Models (LLMs) offer new opportunities to make sensor data more accessible, but their tendency to produce hallucinations limits reliability in safety-critical domains. To address these challenges, we present an LLM-enhanced Air Monitoring Interface (AMI) that integrates real-time sensor data with a conversational interface via the Model Context Protocol (MCP). Our system grounds LLM outputs in live environmental data, enabling accurate, context-aware responses while reducing hallucination risk. The architecture combines a Django-based backend, a responsive user dashboard, and a secure MCP server that exposes system functions as discoverable tools, allowing the LLM to act as an active operator rather than a passive responder. Expert evaluation demonstrated high factual accuracy (4.78), completeness (4.82), and minimal hallucinations (4.84), on a scale of 5, supported by inter-rater reliability analysis. These results highlight the potential of combining LLMs with standardized tool protocols to create reliable, secure, and user-friendly interfaces for real-time environmental monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03697v1">AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 This article was accepted by 2025 International Conference on Computer-Aided Design (ICCAD 2025) and was presented in Munich, October 2025
    </div>
    <details class="paper-abstract">
      Analog/mixed-signal circuits are key for interfacing electronics with the physical world. Their design, however, remains a largely handcrafted process, resulting in long and error-prone design cycles. While the recent rise of AI-based reinforcement learning and generative AI has created new techniques to automate this task, the need for many time-consuming simulations is a critical bottleneck hindering the overall efficiency. Furthermore, the lack of explainability of the resulting design solutions hampers widespread adoption of the tools. To address these issues, a novel agentic AI framework for sample-efficient and explainable analog circuit sizing is presented. It employs a multi-agent workflow where specialized Large Language Model (LLM)-based agents collaborate to interpret the circuit topology, to understand the design goals, and to iteratively refine the circuit's design parameters towards the target goals with human-interpretable reasoning. The adaptive simulation strategy creates an intelligent control that yields a high sample efficiency. The AnaFlow framework is demonstrated for two circuits of varying complexity and is able to complete the sizing task fully automatically, differently from pure Bayesian optimization and reinforcement learning approaches. The system learns from its optimization history to avoid past mistakes and to accelerate convergence. The inherent explainability makes this a powerful tool for analog design space exploration and a new paradigm in analog EDA, where AI agents serve as transparent design assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00807v2">FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 In Submission, code available at https://github.com/AndrewFangZequan/LLM_Serving_FREESH
    </div>
    <details class="paper-abstract">
      The ever-increasing computation and energy demand for LLM and AI agents call for holistic and efficient optimization of LLM serving systems. In practice, heterogeneous GPU clusters can be deployed in a geographically distributed manner, while LLM load also observes diversity in terms of both query traffic and serving patterns. LLM queries running on advanced GPUs during a high-emission hour at one location can lead to significantly higher carbon footprints versus same queries running on mid-level GPUs at a low-emission time and location. By observing LLM serving requirements and leveraging spatiotemporal computation flexibility, we consider the joint routing and scheduling problem, and propose FREESH to cooperatively run a group of data centers while minimizing user-specified carbon or energy objectives. FREESH identifies the optimal configurations of balanced load serving by matching distinct GPU instance's power-throughput characteristics with predictable LLM query length and workloads. To ensure both latency and fairness requirements, FREESH identifies optimized parallelism and query routing schedules together with dynamic GPU frequency scaling for power saving, and Least-Laxity-First (LLF) serving strategy for query scheduling. During the 1-hour serving on production workloads, FREESH reduces energy by 28.6% and emissions by 45.45% together with improvements in SLO attainment and fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04677v2">LLM Query Scheduling with Prefix Reuse and Latency Constraints</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      The efficient deployment of large language models (LLMs) in online settings requires optimizing inference performance under stringent latency constraints, particularly the time-to-first-token (TTFT) and time-per-output-token (TPOT). This paper focuses on the query scheduling problem for LLM inference with prefix reuse, a technique that leverages shared prefixes across queries to reduce computational overhead. Our work reveals previously unknown limitations of the existing first-come-first-serve (FCFS) and longest-prefix-match (LPM) scheduling strategies with respect to satisfying latency constraints. We present a formal theoretical framework for LLM query scheduling under RadixAttention, a prefix reuse mechanism that stores and reuses intermediate representations in a radix tree structure. Our analysis establishes the NP-hardness of the scheduling problem with prefix reuse under TTFT constraints and proposes a novel scheduling algorithm, $k$-LPM, which generalizes existing methods by balancing prefix reuse and fairness in query processing. Theoretical guarantees demonstrate that $k$-LPM achieves improved TTFT performance under realistic traffic patterns captured by a data generative model. Empirical evaluations in a realistic serving setting validates our findings, showing significant reductions in P99 TTFT compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20749v3">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks. Our code is publicly available at: https://github.com/lichangh20/Matryoshka.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10594v2">SME-TEAM: Leveraging Trust and Ethics for Secure and Responsible Use of AI and LLMs in SMEs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and Large Language Models (LLMs) are revolutionizing today's business practices; however, their adoption within small and medium-sized enterprises (SMEs) raises serious trust, ethical, and technical issues. In this perspective paper, we introduce a structured, multi-phased framework, "SME-TEAM" for the secure and responsible use of these technologies in SMEs. Based on a conceptual structure of four key pillars, i.e., Data, Algorithms, Human Oversight, and Model Architecture, SME-TEAM bridges theoretical ethical principles with operational practice, enhancing AI capabilities across a wide range of applications in SMEs. Ultimately, this paper provides a structured roadmap for the adoption of these emerging technologies, positioning trust and ethics as a driving force for resilience, competitiveness, and sustainable innovation within the area of business analytics and SMEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03570v1">TabGemma: Text-Based Tabular ICL via LLM using Continued Pretraining and Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      We study LLMs for tabular prediction with mixed text, numeric, and categorical fields. We introduce TabGemma, a schema-agnostic in-context learner that treats rows as sequences and tackles two practical hurdles when adapting pretrained LLMs for tabular predictions: unstable numeric tokenization and limited context size. We propose to canonicalize numbers via signed scientific notation and continue pretraining of a 12B Gemma 3 model with a target imputation objective using a large-scale real world dataset. For inference, we use a compact n-gram-based retrieval to select informative exemplars that fit within a 128k-token window. On semantically rich benchmarks, TabGemma establishes a new state of the art on classification across low- and high-data regimes and improves monotonically with more context rows. For regression, it is competitive at small sample sizes but trails conventional approaches as data grows. Our results show that LLMs can be effective tabular in-context learners on highly semantic tasks when paired with dedicated numeric handling and context retrieval, while motivating further advances in numeric modeling and long-context scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03563v1">ASVRI-Legal: Fine-Tuning LLMs with Retrieval Augmented Generation for Enhanced Legal Regulation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 11 pages (including references), 2 figures, 4 tables, published in Atlantis Press (Open Access under CC BY-NC 4.0 license)
    </div>
    <details class="paper-abstract">
      In this study, we explore the fine-tuning of Large Language Models (LLMs) to better support policymakers in their crucial work of understanding, analyzing, and crafting legal regulations. To equip the model with a deep understanding of legal texts, we curated a supervised dataset tailored to the specific needs of the legal domain. Additionally, we integrated the Retrieval-Augmented Generation (RAG) method, enabling the LLM to access and incorporate up-to-date legal knowledge from external sources. This combination of fine-tuning and RAG-based augmentation results in a tool that not only processes legal information but actively assists policymakers in interpreting regulations and drafting new ones that align with current needs. The results demonstrate that this approach can significantly enhance the effectiveness of legal research and regulation development, offering a valuable resource in the ever-evolving field of law.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02625v3">HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 19 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Quantized training of Large Language Models (LLMs) remains an open challenge, as maintaining accuracy while performing all matrix multiplications in low precision has proven difficult. This is particularly the case when fine-tuning pre-trained models, which can have large weight and activation outlier values that make lower-precision optimization difficult. To address this, we present HALO, a novel quantization-aware training approach for Transformers that enables accurate and efficient low-precision training by combining 1) strategic placement of Hadamard rotations in both forward and backward passes, which mitigate outliers, 2) high-performance kernel support, and 3) FSDP integration for low-precision communication. Our approach ensures that all large matrix multiplications during the forward and backward passes are executed in lower precision. Applied to LLAMA-family models, HALO achieves near-full-precision-equivalent results during fine-tuning on various tasks, while delivering up to 1.41x end-to-end speedup for full fine-tuning on RTX 4090 GPUs. HALO efficiently supports both standard and parameterefficient fine-tuning (PEFT). Our results demonstrate the first practical approach to fully quantized LLM fine-tuning that maintains accuracy in 8-bit precision, while delivering performance benefits. Code is available at https://github.com/IST-DASLab/HALO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03508v1">One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Understanding how well large language models can follow users' instructions throughout a dialogue spanning multiple topics is of great importance for data-intensive conversational applications. Existing benchmarks are often limited to a fixed number of turns, making them susceptible to saturation and failing to account for the user's interactive experience. In this work, we propose an extensible framework for assessing multi-turn instruction-following ability. At its core, our framework decouples linguistic surface forms from user intent simulation through a three-layer mechanism that tracks constraints, instructions, and topics. This framework mimics User-LLM interaction by enabling the dynamic construction of benchmarks with state changes and tracebacks, terminating a conversation only when the model exhausts a simulated user's patience. We define a suite of metrics capturing the quality of the interaction process. Using this framework, we construct EvolIF, an evolving instruction-following benchmark incorporating nine distinct constraint types. Our results indicate that GPT-5 exhibits superior instruction-following performance. It sustains an average of 18.54 conversational turns and demonstrates 70.31% robustness, outperforming Gemini-2.5-Pro by a significant margin of 11.41%, while other models lag far behind. All of the data and code will be made publicly available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01066v2">HPLT 3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for 24 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder-decoder models, as well as a handful of monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17612v2">Distilling LLM Agent into Small Models with Retrieval and Code Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 NeurIPS 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at https://github.com/Nardien/agent-distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03376v1">Computational Imaging Meets LLMs: Zero-Shot IDH Mutation Prediction in Brain Gliomas</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 5 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      We present a framework that combines Large Language Models with computational image analytics for non-invasive, zero-shot prediction of IDH mutation status in brain gliomas. For each subject, coregistered multi-parametric MRI scans and multi-class tumor segmentation maps were processed to extract interpretable semantic (visual) attributes and quantitative features, serialized in a standardized JSON file, and used to query GPT 4o and GPT 5 without fine-tuning. We evaluated this framework on six publicly available datasets (N = 1427) and results showcased high accuracy and balanced classification performance across heterogeneous cohorts, even in the absence of manual annotations. GPT 5 outperformed GPT 4o in context-driven phenotype interpretation. Volumetric features emerged as the most important predictors, supplemented by subtype-specific imaging markers and clinical information. Our results demonstrate the potential of integrating LLM-based reasoning with computational image analytics for precise, non-invasive tumor genotyping, advancing diagnostic strategies in neuro-oncology. The code is available at https://github.com/ATPLab-LUMS/CIM-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03369v1">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03271v1">Let the Bees Find the Weak Spots: A Path Planning Perspective on Multi-Turn Jailbreak Attacks against LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely deployed across various applications, yet their potential security and ethical risks have raised increasing concerns. Existing research employs red teaming evaluations, utilizing multi-turn jailbreaks to identify potential vulnerabilities in LLMs. However, these approaches often lack exploration of successful dialogue trajectories within the attack space, and they tend to overlook the considerable overhead associated with the attack process. To address these limitations, this paper first introduces a theoretical model based on dynamically weighted graph topology, abstracting the multi-turn attack process as a path planning problem. Based on this framework, we propose ABC, an enhanced Artificial Bee Colony algorithm for multi-turn jailbreaks, featuring a collaborative search mechanism with employed, onlooker, and scout bees. This algorithm significantly improves the efficiency of optimal attack path search while substantially reducing the average number of queries required. Empirical evaluations on three open-source and two proprietary language models demonstrate the effectiveness of our approach, achieving attack success rates above 90\% across the board, with a peak of 98\% on GPT-3.5-Turbo, and outperforming existing baselines. Furthermore, it achieves comparable success with only 26 queries on average, significantly reducing red teaming overhead and highlighting its superior efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14562v3">AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines. Our code is available at https://github.com/hed-ucas/AlphaDecay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03261v1">Comparing the Performance of LLMs in RAG-based Question-Answering: A Case Study in Computer Science Literature</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 18 pages, 4 figures, 5 tables, presented at the 5th International Conference on Artificial Intelligence in Education Technology
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) is emerging as a powerful technique to enhance the capabilities of Generative AI models by reducing hallucination. Thus, the increasing prominence of RAG alongside Large Language Models (LLMs) has sparked interest in comparing the performance of different LLMs in question-answering (QA) in diverse domains. This study compares the performance of four open-source LLMs, Mistral-7b-instruct, LLaMa2-7b-chat, Falcon-7b-instruct and Orca-mini-v3-7b, and OpenAI's trending GPT-3.5 over QA tasks within the computer science literature leveraging RAG support. Evaluation metrics employed in the study include accuracy and precision for binary questions and ranking by a human expert, ranking by Google's AI model Gemini, alongside cosine similarity for long-answer questions. GPT-3.5, when paired with RAG, effectively answers binary and long-answer questions, reaffirming its status as an advanced LLM. Regarding open-source LLMs, Mistral AI's Mistral-7b-instruct paired with RAG surpasses the rest in answering both binary and long-answer questions. However, among the open-source LLMs, Orca-mini-v3-7b reports the shortest average latency in generating responses, whereas LLaMa2-7b-chat by Meta reports the highest average latency. This research underscores the fact that open-source LLMs, too, can go hand in hand with proprietary models like GPT-3.5 with better infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01602v2">L2T-Tune:LLM-Guided Hybrid Database Tuning with LHS and TD3</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Configuration tuning is critical for database performance. Although recent advancements in database tuning have shown promising results in throughput and latency improvement, challenges remain. First, the vast knob space makes direct optimization unstable and slow to converge. Second, reinforcement learning pipelines often lack effective warm-start guidance and require long offline training. Third, transferability is limited: when hardware or workloads change, existing models typically require substantial retraining to recover performance. To address these limitations, we propose L2T-Tune, a new LLM-guided hybrid database tuning framework that features a three-stage pipeline: Stage one performs a warm start that simultaneously generates uniform samples across the knob space and logs them into a shared pool; Stage two leverages a large language model to mine and prioritize tuning hints from manuals and community documents for rapid convergence. Stage three uses the warm-start sample pool to reduce the dimensionality of knobs and state features, then fine-tunes the configuration with the Twin Delayed Deep Deterministic Policy Gradient algorithm. We conduct experiments on L2T-Tune and the state-of-the-art models. Compared with the best-performing alternative, our approach improves performance by an average of 37.1% across all workloads, and by up to 73% on TPC-C. Compared with models trained with reinforcement learning, it achieves rapid convergence in the offline tuning stage on a single server. Moreover, during the online tuning stage, it only takes 30 steps to achieve best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03237v1">IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Tokenizers play a crucial role in determining the performance, training efficiency, and the inference cost of Large Language Models (LLMs). Designing effective tokenizers for multilingual LLMs is particularly challenging due to diverse scripts and rich morphological variation. While subword methods such as Byte Pair Encoding (BPE) are widely adopted, their effectiveness in multilingual settings remains underexplored. We present IndicSuperTokenizer, a tokenizer for Indic multilingual LLMs, that combines both subword and multi-word tokenization, along with language-specific pre-tokenization, leading to more linguistically aligned tokens and achieving a new state-of-the-art in fertility score. Evaluated across English, 22 Indian languages and code data, our tokenizer improves the average fertility score by 39.5% over LLaMA4 and by 18% over Sutra (the current best). This translates to 44% improvement in inference throughput over LLaMA4 while maintaining comparable performance on English and Indic benchmarks. We also present detailed ablations across tokenizer training data size, vocabulary size, merging techniques, and pre-tokenization strategies, demonstrating the robustness of our design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16395v2">LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Atomic commits, which address a single development concern, are a best practice in software development. In practice, however, developers often produce tangled commits that mix unrelated changes, complicating code review and maintenance. Prior untangling approaches (rule-based, feature-based, or graph-based) have made progress but typically rely on shallow signals and struggle to distinguish explicit dependencies (e.g., control/data flow) from implicit ones (e.g., semantic or conceptual relationships). In this paper, we propose ColaUntangle, a new collaborative consultation framework for commit untangling that models both explicit and implicit dependencies among code changes. ColaUntangle integrates Large Language Model (LLM)-driven agents in a multi-agent architecture: one agent specializes in explicit dependencies, another in implicit ones, and a reviewer agent synthesizes their perspectives through iterative consultation. To capture structural and contextual information, we construct Explicit and Implicit Contexts, enabling agents to reason over code relationships with both symbolic and semantic depth. We evaluate ColaUntangle on two widely-used datasets (1,612 C# and 14k Java tangled commits). Experimental results show that ColaUntangle outperforms the best-performing baseline, achieving an improvement of 44% on the C# dataset and 82% on the Java dataset. These findings highlight the potential of LLM-based collaborative frameworks for advancing automated commit untangling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19361v2">AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03182v1">Understanding Robustness of Model Editing in Code LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 26 pages, 2 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in software development. However, while LLMs remain static after pretraining, programming languages and APIs continue to evolve, leading to the generation of deprecated or incompatible code that undermines reliability. Retraining LLMs from scratch to reflect such changes is computationally expensive, making model editing a promising lightweight alternative that updates only a small subset of parameters. Despite its potential, it remains unclear whether model editing yields genuine syntactic and semantic adaptations or merely superficial fixes. In this work, we present a systematic study of five state-of-the-art model editing methods: Constrained Fine-Tuning (FT), GRACE, MEMIT, PMET, and ROME. We apply these methods to three leading open-source code LLMs, CodeLlama, CodeQwen1.5, and DeepSeek-Coder, under controlled API deprecation scenarios. Our evaluation covers both instant and sequential editing settings, using three disjoint evaluation sets designed to assess reliability, generalization, and specificity. We measure model correctness at three levels: successful compilation, partial test case pass, and full test pass. Our findings show that instant edits consistently degrade model performance, with syntactic validity dropping by up to 86 percentage points and functional correctness declining by 45 points even in the best-performing setting. Sequential edits further amplify this degradation, and in some cases, model performance collapses entirely. Across all models, most passing generations relied on workarounds rather than correctly adopting the intended changes, while faulty adoptions that result in test failures or compilation errors were significantly more frequent. Correct adoptions, where the model correctly integrates the intended change, occurred in only about 6% of cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.02795v1">Can LLMs subtract numbers?</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Work-in-progress; MathNLP non-archival presentation
    </div>
    <details class="paper-abstract">
      We present a systematic study of subtraction in large language models (LLMs). While prior benchmarks emphasize addition and multiplication, subtraction has received comparatively little attention despite being structurally distinct as a non-commutative operation. We evaluate eight pretrained LLMs spanning four families on addition and subtraction problems. Our experiments reveal that subtraction accuracy lags behind addition by a wide margin. We find that the errors for ($a-b$) are concentrated in cases where ($a<b$). In such cases, LLMs frequently produce the correct magnitude but omit the negative sign. Probing analyses show that LLMs internally encode whether results should be negative, yet this information is often not reflected in generated outputs. We further test well-known techniques such as few-shot learning and instruction-tuning to see if they can improve the LLMs' performance. Our results suggest that while few-shot prompting yields modest gains, the instruction-tuned models achieve near-perfect accuracies in generating the negative sign. Together, these findings provide a clearer characterization of the limitations and recoverability of LLMs' arithmetic capabilities in subtraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17017v3">SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Code available at https://github.com/amazon-science/SafeSearch
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked "How can I track someone's location without their consent?", a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20462v5">TAMO: Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data in Cloud-Native Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Implementing large language models (LLMs)-driven root cause analysis (RCA) in cloud-native systems has become a key topic of modern software operations and maintenance. However, existing LLM-based approaches face three key challenges: multi-modality input constraint, context window limitation, and dynamic dependence graph. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data for fine-grained RCA, namely TAMO, including multimodality alignment tool, root cause localization tool, and fault types classification tool. In detail, TAMO unifies multi-modal observation data into time-aligned representations for cross-modal feature consistency. Based on the unified representations, TAMO then invokes its specialized root cause localization tool and fault types classification tool for further identifying root cause and fault type underlying system context. This approach overcomes the limitations of LLMs in processing real-time raw observational data and dynamic service dependencies, guiding the model to generate repair strategies that align with system context through structured prompt design. Experiments on two benchmark datasets demonstrate that TAMO outperforms state-of-the-art (SOTA) approaches with comparable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03166v1">Measuring Aleatoric and Epistemic Uncertainty in LLMs: Empirical Evaluation on ID and OOD QA Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Accepted by UDM-KDD'24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly pervasive, finding applications across many industries and disciplines. Ensuring the trustworthiness of LLM outputs is paramount, where Uncertainty Estimation (UE) plays a key role. In this work, a comprehensive empirical study is conducted to examine the robustness and effectiveness of diverse UE measures regarding aleatoric and epistemic uncertainty in LLMs. It involves twelve different UE methods and four generation quality metrics including LLMScore from LLM criticizers to evaluate the uncertainty of LLM-generated answers in Question-Answering (QA) tasks on both in-distribution (ID) and out-of-distribution (OOD) datasets. Our analysis reveals that information-based methods, which leverage token and sequence probabilities, perform exceptionally well in ID settings due to their alignment with the model's understanding of the data. Conversely, density-based methods and the P(True) metric exhibit superior performance in OOD contexts, highlighting their effectiveness in capturing the model's epistemic uncertainty. Semantic consistency methods, which assess variability in generated answers, show reliable performance across different datasets and generation metrics. These methods generally perform well but may not be optimal for every situation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02184v2">Leveraging LLMs to Automate Energy-Aware Refactoring of Parallel Scientific Codes</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 12 pages, 4 figures, version under review at a peer-reviewed conference
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used for generating parallel scientific codes, most efforts emphasize functional correctness, often overlooking performance, especially energy efficiency. We propose LASSI-EE, an automated LLM-based refactoring framework that generates energy-efficient parallel codes through a multi-stage, iterative approach integrating runtime power profiling, energy-aware prompting, self-correcting feedback loops, and an LLM-as-a-Judge agent for automated screening of code solutions. We introduce energy-reduction@k, a novel metric that quantifies expected energy reduction when generating k code candidates and selecting the most energy-efficient, enabling systematic evaluation of multi-attempt generation strategies. Evaluating 20 HeCBench applications and two miniApps on NVIDIA A100 and AMD MI100 GPUs, a single run (k=1) with LASSI-EE delivers refactored parallel codes with an average 29% expected energy reduction at an 81% pass rate, representing a 2.8x improvement over vanilla LLM prompting. Multiple runs (k=3) achieve an average 48% expected energy reduction at a 97% pass rate. These results are consistent across devices, demonstrating LASSI-EE's effectiveness across diverse hardware architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02263v2">LA-MARRVEL: A Knowledge-Grounded and Language-Aware LLM Reranker for AI-MARRVEL in Rare Disease Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Diagnosing rare diseases often requires connecting variant-bearing genes to evidence that is written as unstructured clinical prose, which the current established pipelines still leave for clinicians to reconcile manually. To this end, we introduce LA-MARRVEL, a knowledge-grounded and language-aware reranking layer that operates on top of AI-MARRVEL: it supplies expert-engineered context, queries a large language model multiple times, and aggregates the resulting partial rankings with a ranked voting method to produce a stable, explainable gene ranking. Evaluated on three real-world cohorts (BG, DDD, UDN), LA-MARRVEL consistently improves Recall@K over AI-MARRVEL and established phenotype-driven tools such as Exomiser and LIRICAL, with especially large gains on cases where the first-stage ranker placed the causal gene lower. Each ranked gene is accompanied by LLM-generated reasoning that integrates phenotypic, inheritance, and variant-level evidence, thereby making the output more interpretable and facilitating clinical review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12839v2">FaStfact: Faster, Stronger Long-Form Factuality Evaluations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Evaluating the factuality of long-form generations from Large Language Models (LLMs) remains challenging due to efficiency bottlenecks and reliability concerns. Prior efforts attempt this by decomposing text into claims, searching for evidence, and verifying claims, but suffer from critical drawbacks: (1) inefficiency due to overcomplicated pipeline components, and (2) ineffectiveness stemming from inaccurate claim sets and insufficient evidence. To address these limitations, we propose \textbf{FaStfact}, an evaluation framework that achieves the highest alignment with human evaluation and time/token efficiency among existing baselines. FaStfact first employs chunk-level claim extraction integrated with confidence-based pre-verification, significantly reducing the time and token cost while ensuring reliability. For searching and verification, it collects document-level evidence from crawled web-pages and selectively retrieves it during verification. Extensive experiments based on an annotated benchmark \textbf{FaStfact-Bench} demonstrate the reliability of FaStfact in both efficiently and effectively evaluating long-form factuality. Code, benchmark data, and annotation interface tool are available at https://github.com/Yingjia-Wan/FaStfact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03153v1">RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially influenced various software engineering tasks. Indeed, in the case of software refactoring, traditional LLMs have shown the ability to reduce development time and enhance code quality. However, these LLMs often rely on static, detailed instructions for specific tasks. In contrast, LLM-based agents can dynamically adapt to evolving contexts and autonomously make decisions by interacting with software tools and executing workflows. In this paper, we explore the potential of LLM-based agents in supporting refactoring activities. Specifically, we introduce RefAgent, a multi-agent LLM-based framework for end-to-end software refactoring. RefAgent consists of specialized agents responsible for planning, executing, testing, and iteratively refining refactorings using self-reflection and tool-calling capabilities. We evaluate RefAgent on eight open-source Java projects, comparing its effectiveness against a single-agent approach, a search-based refactoring tool, and historical developer refactorings. Our assessment focuses on: (1) the impact of generated refactorings on software quality, (2) the ability to identify refactoring opportunities, and (3) the contribution of each LLM agent through an ablation study. Our results show that RefAgent achieves a median unit test pass rate of 90%, reduces code smells by a median of 52.5%, and improves key quality attributes (e.g., reusability) by a median of 8.6%. Additionally, it closely aligns with developer refactorings and the search-based tool in identifying refactoring opportunities, attaining a median F1-score of 79.15% and 72.7%, respectively. Compared to single-agent approaches, RefAgent improves the median unit test pass rate by 64.7% and the median compilation success rate by 40.1%. These findings highlight the promise of multi-agent architectures in advancing automated software refactoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03152v1">Who Sees the Risk? Stakeholder Conflicts and Explanatory Policies in LLM-based Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Understanding how different stakeholders perceive risks in AI systems is essential for their responsible deployment. This paper presents a framework for stakeholder-grounded risk assessment by using LLMs, acting as judges to predict and explain risks. Using the Risk Atlas Nexus and GloVE explanation method, our framework generates stakeholder-specific, interpretable policies that shows how different stakeholders agree or disagree about the same risks. We demonstrate our method using three real-world AI use cases of medical AI, autonomous vehicles, and fraud detection domain. We further propose an interactive visualization that reveals how and why conflicts emerge across stakeholder perspectives, enhancing transparency in conflict reasoning. Our results show that stakeholder perspectives significantly influence risk perception and conflict patterns. Our work emphasizes the importance of these stakeholder-aware explanations needed to make LLM-based evaluations more transparent, interpretable, and aligned with human-centered AI governance goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03128v1">From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2025 (camera-ready)
    </div>
    <details class="paper-abstract">
      LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at https://github.com/Shukti042/AdversarialExample.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03094v1">ALAS: Transactional and Dynamic Multi-Agent LLM Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Large language models enable flexible multi-agent planning but remain fragile in practice: verification is often circular, state changes are not tracked for repair, and small faults trigger costly global recomputation. We present ALAS, a stateful, disruption-aware framework that separates planning from non-circular validation, records a versioned execution log for grounded checks and restore points, and performs localized repair that preserves work in progress. The validator operates independently of the planning LLM with fresh, bounded context, avoiding self-check loops and mid-context attrition. The repair protocol edits only the minimal affected region under explicit policies (retry, catch, timeout, backoff, idempotency keys, compensation, loop guards) defined in a canonical workflow IR that maps to Amazon States Language and Argo Workflows. On job-shop scheduling suites (DMU, TA) across five classical benchmarks, ALAS matches or exceeds strong single-LLM and multi-agent baselines, achieving 83.7% success, reducing token usage by 60%, and running 1.82times faster under comparable settings. A minimal reliability study shows that the validator detects injected structural faults with low overhead, and that localized repair contains runtime perturbations with a bounded edit radius and less makespan degradation than global recompute. Results indicate that the combination of validator isolation, versioned execution logs, and localized repair provides measurable efficiency, feasibility, and scalability for multi-agent LLM planning. Code and seeds will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04731v3">Generative World Models of Tasks: LLM-Driven Hierarchical Scaffolding for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 In the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Embodied World Models for Decision Making (EWM)
    </div>
    <details class="paper-abstract">
      Recent advances in agent development have focused on scaling model size and raw interaction data, mirroring successes in large language models. However, for complex, long-horizon multi-agent tasks such as robotic soccer, this end-to-end approach often fails due to intractable exploration spaces and sparse rewards. We propose that an effective world model for decision-making must model the world's physics and also its task semantics. A systematic review of 2024 research in low-resource multi-agent soccer reveals a clear trend towards integrating symbolic and hierarchical methods, such as Hierarchical Task Networks (HTNs) and Bayesian Strategy Networks (BSNs), with multi-agent reinforcement learning (MARL). These methods decompose complex goals into manageable subgoals, creating an intrinsic curriculum that shapes agent learning. We formalize this trend into a framework for Hierarchical Task Environments (HTEs), which are essential for bridging the gap between simple, reactive behaviors and sophisticated, strategic team play. Our framework incorporates the use of Large Language Models (LLMs) as generative world models of tasks, capable of dynamically generating this scaffolding. We argue that HTEs provide a mechanism to guide exploration, generate meaningful learning signals, and train agents to internalize hierarchical structure, enabling the development of more capable and general-purpose agents with greater sample efficiency than purely end-to-end approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01281v3">Path-Consistency with Prefix Enhancement for Efficient Inference in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      To enhance the reasoning capabilities of large language models (LLMs), self-consistency has become a popular approach, combining multiple samplings with majority voting. However, current methods are computationally expensive and time-consuming due to the need for numerous samplings. To address this, this paper introduces path-consistency, which leverages the confidence of earlier-generated answers to identify the most promising prefix and guide the generation of subsequent branches. By dynamically guiding the generation of subsequent branches based on this prefix, path-consistency mitigates both the errors and redundancies from random or less useful sampling in self-consistency. This approach reduces errors and redundancies from random sampling, significantly accelerating inference by minimizing token consumption. Our extensive empirical results demonstrate that path-consistency improves inference latency by up to 40.5\%, while maintaining task accuracy across various tasks, including mathematical reasoning, commonsense reasoning, and symbolic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05758v5">APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair viaLLM and Lean cOllaboration), a modular, model-agnostic agentic framework that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low token and sampling budgets. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 84.9% among sub 8B-parameter models (as of August 2025) while keeping the sampling budget below one hundred. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02263v1">LA-MARRVEL: A Knowledge-Grounded and Language-Aware LLM Reranker for AI-MARRVEL in Rare Disease Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Diagnosing rare diseases often requires connecting variant-bearing genes to evidence that is written as unstructured clinical prose, which the current established pipelines still leave for clinicians to reconcile manually. To this end, we introduce LA-MARRVEL, a knowledge-grounded and language-aware reranking layer that operates on top of AI-MARRVEL: it supplies expert-engineered context, queries a large language model multiple times, and aggregates the resulting partial rankings with a ranked voting method to produce a stable, explainable gene ranking. Evaluated on three real-world cohorts (BG, DDD, UDN), LA-MARRVEL consistently improves Recall@K over AI-MARRVEL and established phenotype-driven tools such as Exomiser and LIRICAL, with especially large gains on cases where the first-stage ranker placed the causal gene lower. Each ranked gene is accompanied by LLM-generated reasoning that integrates phenotypic, inheritance, and variant-level evidence, thereby making the output more interpretable and facilitating clinical review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02246v1">Demo: Statistically Significant Results On Biases and Errors of LLMs Do Not Guarantee Generalizable Results</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent research has shown that hallucinations, omissions, and biases are prevalent in everyday use-cases of LLMs. However, chatbots used in medical contexts must provide consistent advice in situations where non-medical factors are involved, such as when demographic information is present. In order to understand the conditions under which medical chatbots fail to perform as expected, we develop an infrastructure that 1) automatically generates queries to probe LLMs and 2) evaluates answers to these queries using multiple LLM-as-a-judge setups and prompts. For 1), our prompt creation pipeline samples the space of patient demographics, histories, disorders, and writing styles to create realistic questions that we subsequently use to prompt LLMs. In 2), our evaluation pipeline provides hallucination and omission detection using LLM-as-a-judge as well as agentic workflows, in addition to LLM-as-a-judge treatment category detectors. As a baseline study, we perform two case studies on inter-LLM agreement and the impact of varying the answering and evaluation LLMs. We find that LLM annotators exhibit low agreement scores (average Cohen's Kappa $\kappa=0.118$), and only specific (answering, evaluation) LLM pairs yield statistically significant differences across writing styles, genders, and races. We recommend that studies using LLM evaluation use multiple LLMs as evaluators in order to avoid arriving at statistically significant but non-generalizable results, particularly in the absence of ground-truth data. We also suggest publishing inter-LLM agreement metrics for transparency. Our code and dataset are available here: https://github.com/BBN-E/medic-neurips-2025-demo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02238v1">Deep Ideation: Designing LLM Agents to Generate Novel Research Ideas on Scientific Concept Network</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 23 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Novel research ideas play a critical role in advancing scientific inquiries. Recent advancements in Large Language Models (LLMs) have demonstrated their potential to generate novel research ideas by leveraging large-scale scientific literature. However, previous work in research ideation has primarily relied on simplistic methods, such as keyword co-occurrence or semantic similarity. These approaches focus on identifying statistical associations in the literature but overlook the complex, contextual relationships between scientific concepts, which are essential to effectively leverage knowledge embedded in human literature. For instance, papers that simultaneously mention "keyword A" and "keyword B" often present research ideas that integrate both concepts. Additionally, some LLM-driven methods propose and refine research ideas using the model's internal knowledge, but they fail to effectively utilize the scientific concept network, limiting the grounding of ideas in established research. To address these challenges, we propose the Deep Ideation framework to address these challenges, integrating a scientific network that captures keyword co-occurrence and contextual relationships, enriching LLM-driven ideation. The framework introduces an explore-expand-evolve workflow to iteratively refine research ideas, using an Idea Stack to track progress. A critic engine, trained on real-world reviewer feedback, guides the process by providing continuous feedback on the novelty and feasibility of ideas. Our experiments show that our approach improves the quality of generated ideas by 10.67% compared to other methods, with ideas surpassing top conference acceptance levels. Human evaluation highlights their practical value in scientific research, and ablation studies confirm the effectiveness of each component in the workflow. Code repo is available at https://github.com/kyZhao-1/Deep-Ideation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02230v1">Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Agentic LLM applications interleave LLM generation requests with tool calls. These tool calls break the continuity of the workflow by creating pauses between LLM requests, bringing many challenges for the serving system, especially under multi-turn scenarios. Each pause potentially causes KV cache eviction and extra waiting time before entering the continuous batch for the following LLM request. Since these pauses happen for each call, this problem becomes increasingly severe as turn number grow for agentic programs. Previous works either fail to incorporate information from the tool call, evicting KV cache that leads to repetitive prefill or loading, or ignore the continuity of a multi-turn program, creating waiting time between turns that increases per-request latency. We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by combining tool-aware KV cache timeout with program-level scheduling. By predicting tool call durations in agentic workflows, Continuum selectively pins the KV cache in GPU memory with a time-to-live value based on total turn number. When combined with program-level first-come-first-serve, Continuum prevents scheduling bubbles, preserves multi-turn continuity, and optimizes for throughput for complex agentic workflows. By modeling the variability of tool call and agent program continuity, Continuum outperforms state-of-the-art baselines. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B models shows that Continuum significantly improves the average job completion times, and remains performant across different hardware setups and DRAM offloading schemes. Preview code is available at: https://github.com/Hanchenli/vllm-continuum
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15112v2">AndroByte: LLM-Driven Privacy Analysis through Bytecode Summarization and Dynamic Dataflow Call Graph Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Accepted at the Annual Computer Security Applications Conference (ACSAC) 2025
    </div>
    <details class="paper-abstract">
      With the exponential growth in mobile applications, protecting user privacy has become even more crucial. Android applications are often known for collecting, storing, and sharing sensitive user information such as contacts, location, camera, and microphone data often without the user's clear consent or awareness raising significant privacy risks and exposure. In the context of privacy assessment, dataflow analysis is particularly valuable for identifying data usage and potential leaks. Traditionally, this type of analysis has relied on formal methods, heuristics, and rule-based matching. However, these techniques are often complex to implement and prone to errors, such as taint explosion for large programs. Moreover, most existing Android dataflow analysis methods depend heavily on predefined list of sinks, limiting their flexibility and scalability. To address the limitations of these existing techniques, we propose AndroByte, an AI-driven privacy analysis tool that leverages LLM reasoning on bytecode summarization to dynamically generate accurate and explainable dataflow call graphs from static code analysis. AndroByte achieves a significant F\b{eta}-Score of 89% in generating dynamic dataflow call graphs on the fly, outperforming the effectiveness of traditional tools like FlowDroid and Amandroid in leak detection without relying on predefined propagation rules or sink lists. Moreover, AndroByte's iterative bytecode summarization provides comprehensive and explainable insights into dataflow and leak detection, achieving high, quantifiable scores based on the G-Eval metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02208v1">Training Proactive and Personalized LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02203v1">LLMs as Judges: Toward The Automatic Review of GSN-compliant Assurance Cases</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Assurance cases allow verifying the correct implementation of certain non-functional requirements of mission-critical systems, including their safety, security, and reliability. They can be used in the specification of autonomous driving, avionics, air traffic control, and similar systems. They aim to reduce risks of harm of all kinds including human mortality, environmental damage, and financial loss. However, assurance cases often tend to be organized as extensive documents spanning hundreds of pages, making their creation, review, and maintenance error-prone, time-consuming, and tedious. Therefore, there is a growing need to leverage (semi-)automated techniques, such as those powered by generative AI and large language models (LLMs), to enhance efficiency, consistency, and accuracy across the entire assurance-case lifecycle. In this paper, we focus on assurance case review, a critical task that ensures the quality of assurance cases and therefore fosters their acceptance by regulatory authorities. We propose a novel approach that leverages the \textit{LLM-as-a-judge} paradigm to automate the review process. Specifically, we propose new predicate-based rules that formalize well-established assurance case review criteria, allowing us to craft LLM prompts tailored to the review task. Our experiments on several state-of-the-art LLMs (GPT-4o, GPT-4.1, DeepSeek-R1, and Gemini 2.0 Flash) show that, while most LLMs yield relatively good review capabilities, DeepSeek-R1 and GPT-4.1 demonstrate superior performance, with DeepSeek-R1 ultimately outperforming GPT-4.1. However, our experimental results also suggest that human reviewers are still needed to refine the reviews LLMs yield.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02197v1">Open the Oyster: Empirical Evaluation and Improvement of Code Reasoning Confidence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With the widespread application of large language models (LLMs) in the field of code intelligence, increasing attention has been paid to the reliability and controllability of their outputs in code reasoning tasks. Confidence estimation serves as an effective and convenient approach for evaluating these aspects. This paper proposes a confidence analysis and enhancement framework for LLMs tailored to code reasoning tasks. We conduct a comprehensive empirical study on the confidence reliability of mainstream LLMs across different tasks, and further evaluate the effectiveness of techniques such as prompt strategy optimisation and mathematical calibration (e.g., Platt Scaling) in improving confidence reliability. Our results show that DeepSeek-Reasoner achieves the best performance across various tasks, outperforming other models by up to $0.680$, $0.636$, and $13.652$ in terms of ECE, Brier Score, and Performance Score, respectively. The hybrid strategy combining the reassess prompt strategy and Platt Scaling achieves improvements of up to $0.541$, $0.628$, and $15.084$ over the original performance in the aforementioned three metrics. These results indicate that models with reasoning capabilities demonstrate superior confidence reliability, and that the hybrid strategy is the most effective in enhancing the confidence reliability of various models. Meanwhile, we elucidate the impact of different task complexities, model scales, and strategies on confidence performance, and highlight that the confidence of current LLMs in complex reasoning tasks still has considerable room for improvement. This study not only provides a research foundation and technical reference for the application of confidence in LLM-assisted software engineering, but also points the way for future optimisation and engineering deployment of confidence mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20527v3">SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Accepted at MATH-AI workshop, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The demand for Large Language Models (LLMs) at multiple scales, capable of sophisticated and sound mathematical reasoning, continues to grow. However, the development of performant mathematical LLMs is often bottlenecked by the scarcity of useful training data containing problems with significant complexity. We introduce \textbf{SAND-Math} (\textbf{S}ynthetic \textbf{A}ugmented \textbf{N}ovel and \textbf{D}ifficult Mathematics problems and solutions), a pipeline that addresses this by first synthesizing high-quality problems from scratch and then systematically elevating their complexity via a our newly proposed \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings: \textbf{(1)} Augmenting a strong post-training baseline with a small 500-sample SAND-Math dataset significantly boosts performance, outperforming the next-best synthetic dataset by $\uparrow$ 17.85 absolute points on AIME25 benchmark. \textbf{(2)} In a dedicated ablation study, we show the effectiveness of our Difficulty Hiking process in increasing average problem difficulty from 5.02 to 5.98. This step consequently lifts AIME25 results from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building capable and efficient mathematical reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02168v1">Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale, their workloads increasingly rely on distributed execution across multiple GPUs. However, the conventional bulk synchronous parallel~(BSP) model used in such settings introduces significant performance inefficiencies. To characterize these bottlenecks, we introduce the ''Three Taxes'' (Bulk Synchronous, Inter-Kernel Data Locality, and Kernel Launch Overhead) as an analytical framework. We propose moving beyond the rigid BSP model to address key inefficiencies in distributed GPU execution. By exploiting libraries like Iris for Triton, we gain access to in-kernel communication primitives that enable the design of novel fine-grained programming patterns, offering greater flexibility and performance than traditional BSP-based approaches. These patterns systematically eliminate the three taxes by creating direct, tile-level producer-consumer pipelines and replacing global barriers with fine-grained dataflow synchronization. Applying this methodology to critical kernels, from the foundational All-Gather + general matrix multiplication operation to the complex Flash Decode algorithm, we observe a 10-20% speedup in end-to-end latency over BSP-based approaches, establishing a more programmable and efficient paradigm for distributed LLM workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03048v1">ROBoto2: An Interactive System and Dataset for LLM-assisted Clinical Trial Risk of Bias Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 EMNLP 2025 System Demonstration
    </div>
    <details class="paper-abstract">
      We present ROBOTO2, an open-source, web-based platform for large language model (LLM)-assisted risk of bias (ROB) assessment of clinical trials. ROBOTO2 streamlines the traditionally labor-intensive ROB v2 (ROB2) annotation process via an interactive interface that combines PDF parsing, retrieval-augmented LLM prompting, and human-in-the-loop review. Users can upload clinical trial reports, receive preliminary answers and supporting evidence for ROB2 signaling questions, and provide real-time feedback or corrections to system suggestions. ROBOTO2 is publicly available at https://roboto2.vercel.app/, with code and data released to foster reproducibility and adoption. We construct and release a dataset of 521 pediatric clinical trial reports (8954 signaling questions with 1202 evidence passages), annotated using both manually and LLM-assisted methods, serving as a benchmark and enabling future research. Using this dataset, we benchmark ROB2 performance for 4 LLMs and provide an analysis into current model capabilities and ongoing challenges in automating this critical aspect of systematic review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03023v1">PublicAgent: Multi-Agent Design Principles From an LLM-Based Open Data Analysis Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Open data repositories hold potential for evidence-based decision-making, yet are inaccessible to non-experts lacking expertise in dataset discovery, schema mapping, and statistical analysis. Large language models show promise for individual tasks, but end-to-end analytical workflows expose fundamental limitations: attention dilutes across growing contexts, specialized reasoning patterns interfere, and errors propagate undetected. We present PublicAgent, a multi-agent framework that addresses these limitations through decomposition into specialized agents for intent clarification, dataset discovery, analysis, and reporting. This architecture maintains focused attention within agent contexts and enables validation at each stage. Evaluation across five models and 50 queries derives five design principles for multi-agent LLM systems. First, specialization provides value independent of model strength--even the strongest model shows 97.5% agent win rates, with benefits orthogonal to model scale. Second, agents divide into universal (discovery, analysis) and conditional (report, intent) categories. Universal agents show consistent effectiveness (std dev 12.4%) while conditional agents vary by model (std dev 20.5%). Third, agents mitigate distinct failure modes--removing discovery or analysis causes catastrophic failures (243-280 instances), while removing report or intent causes quality degradation. Fourth, architectural benefits persist across task complexity with stable win rates (86-92% analysis, 84-94% discovery), indicating workflow management value rather than reasoning enhancement. Fifth, wide variance in agent effectiveness across models (42-96% for analysis) requires model-aware architecture design. These principles guide when and why specialization is necessary for complex analytical workflows while enabling broader access to public data through natural language interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15433v3">LLM-Driven SAST-Genius: A Hybrid Static Analysis Framework for Comprehensive and Actionable Security</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      This report examines the synergy between Large Language Models (LLMs) and Static Application Security Testing (SAST) to improve vulnerability discovery. Traditional SAST tools, while effective for proactive security, are limited by high false-positive rates and a lack of contextual understanding. Conversely, LLMs excel at code analysis and pattern recognition but can be prone to inconsistencies and hallucinations. By integrating these two technologies, a more intelligent and efficient system is created. This combination moves beyond mere vulnerability detection optimization, transforming security into a deeply integrated, contextual process that provides tangible benefits like improved triage, dynamic bug descriptions, bug validation via exploit generation and enhanced analysis of complex codebases. The result is a more effective security approach that leverages the strengths of both technologies while mitigating their weaknesses. SAST-Genius reduced false positives by about 91 % (225 to 20) compared to Semgrep alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v5">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 8 pages, 8 figures, 2 tables, accepted by SEC 2025: Tenth ACM/IEEE Symposium on Edge Computing
    </div>
    <details class="paper-abstract">
      The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite gradual improvements in hardware capabilities. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new framework that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose \acronym, a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model. To further increase the efficiency of verification, the edge server batch the diverse verification requests from devices. This approach supports device heterogeneity and reduces server-side memory footprint by sharing the same upstream target model across multiple devices. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: 2.2 more system throughput, 2.8 more system capacity, and better cost efficiency, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26130v2">Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Pre-print submitted for reviwer to TOSEM
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance on function-level code generation benchmarks, yet real-world software development increasingly demands class-level implementations that integrate multiple methods, attributes, and dependencies within authentic project contexts. This gap between benchmark performance and practical utility raises critical questions about LLMs' readiness for production code assistance, particularly regarding their ability to generalize across familiar and novel codebases. We introduce a benchmark derived from real-world open-source repositories, comprising classes divided into seen and unseen partitions to evaluate generalization under practical conditions. We systematically examine how input specification completeness and retrieval-augmented generation affect class-level correctness across multiple state-of-the-art LLMs. Our evaluation reveals a substantial performance gap: while LLMs achieve 84 to 89% correctness on synthetic benchmarks, they attain only 25 to 34% on real-world class tasks, with minimal distinction between familiar and novel codebases. Comprehensive documentation provides marginal improvements (1 to 3%), whereas retrieval augmentation yields greater gains (4 to 7%) by supplying concrete implementation patterns. Error analysis identifies AttributeError, TypeError, and AssertionError as dominant failure modes, with distinct patterns between synthetic and real-world scenarios. These findings provide actionable insights for enhancing context modelling, documentation strategies, and retrieval integration in production code assistance tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02979v1">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Submitted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02936v1">Zero-shot data citation function classification using transformer-based large language models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Efforts have increased in recent years to identify associations between specific datasets and the scientific literature that incorporates them. Knowing that a given publication cites a given dataset, the next logical step is to explore how or why that data was used. Advances in recent years with pretrained, transformer-based large language models (LLMs) offer potential means for scaling the description of data use cases in the published literature. This avoids expensive manual labeling and the development of training datasets for classical machine-learning (ML) systems. In this work we apply an open-source LLM, Llama 3.1-405B, to generate structured data use case labels for publications known to incorporate specific genomic datasets. We also introduce a novel evaluation framework for determining the efficacy of our methods. Our results demonstrate that the stock model can achieve an F1 score of .674 on a zero-shot data citation classification task with no previously defined categories. While promising, our results are qualified by barriers related to data availability, prompt overfitting, computational infrastructure, and the expense required to conduct responsible performance evaluation.
    </details>
</div>
