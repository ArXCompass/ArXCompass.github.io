# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00651v1">Open-Source LLM-Driven Federated Transformer for Predictive IoV Management</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The proliferation of connected vehicles within the Internet of Vehicles (IoV) ecosystem presents critical challenges in ensuring scalable, real-time, and privacy-preserving traffic management. Existing centralized IoV solutions often suffer from high latency, limited scalability, and reliance on proprietary Artificial Intelligence (AI) models, creating significant barriers to widespread deployment, particularly in dynamic and privacy-sensitive environments. Meanwhile, integrating Large Language Models (LLMs) in vehicular systems remains underexplored, especially concerning prompt optimization and effective utilization in federated contexts. To address these challenges, we propose the Federated Prompt-Optimized Traffic Transformer (FPoTT), a novel framework that leverages open-source LLMs for predictive IoV management. FPoTT introduces a dynamic prompt optimization mechanism that iteratively refines textual prompts to enhance trajectory prediction. The architecture employs a dual-layer federated learning paradigm, combining lightweight edge models for real-time inference with cloud-based LLMs to retain global intelligence. A Transformer-driven synthetic data generator is incorporated to augment training with diverse, high-fidelity traffic scenarios in the Next Generation Simulation (NGSIM) format. Extensive evaluations demonstrate that FPoTT, utilizing EleutherAI Pythia-1B, achieves 99.86% prediction accuracy on real-world data while maintaining high performance on synthetic datasets. These results underscore the potential of open-source LLMs in enabling secure, adaptive, and scalable IoV management, offering a promising alternative to proprietary solutions in smart mobility ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08067v4">Reward-Augmented Data Enhances Direct Preference Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Preference alignment in Large Language Models (LLMs) has significantly improved their ability to adhere to human instructions and intentions. However, existing direct alignment algorithms primarily focus on relative preferences and often overlook the qualitative aspects of responses, despite having access to preference data that includes reward scores from judge models during AI feedback. Striving to maximize the implicit reward gap between the chosen and the slightly inferior rejected responses can cause overfitting and unnecessary unlearning of the high-quality rejected responses. The unawareness of the reward scores also drives the LLM to indiscriminately favor the low-quality chosen responses and fail to generalize to optimal responses that are sparse in data. To overcome these shortcomings, our study introduces reward-conditioned LLM policies that discern and learn from the entire spectrum of response quality within the dataset, helping extrapolate to more optimal regions. We propose an effective yet simple data relabeling method that conditions the preference pairs on quality scores to construct a reward-augmented dataset. The experiments across various benchmarks and diverse models demonstrate that our approach consistently boosts DPO by a considerable margin. Through comprehensive ablation studies, we demonstrate that our method not only maximizes the utility of preference data but also mitigates the issue of unlearning, demonstrating its broad effectiveness beyond mere data expansion. Our code is available at https://github.com/shenao-zhang/reward-augmented-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00626v1">The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them)</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00610v1">Combining LLMs with Logic-Based Framework to Explain MCTS</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Accepted by AAMAS-25 as an extended abstract
    </div>
    <details class="paper-abstract">
      In response to the lack of trust in Artificial Intelligence (AI) for sequential planning, we design a Computational Tree Logic-guided large language model (LLM)-based natural language explanation framework designed for the Monte Carlo Tree Search (MCTS) algorithm. MCTS is often considered challenging to interpret due to the complexity of its search trees, but our framework is flexible enough to handle a wide range of free-form post-hoc queries and knowledge-based inquiries centered around MCTS and the Markov Decision Process (MDP) of the application domain. By transforming user queries into logic and variable statements, our framework ensures that the evidence obtained from the search tree remains factually consistent with the underlying environmental dynamics and any constraints in the actual stochastic control process. We evaluate the framework rigorously through quantitative assessments, where it demonstrates strong performance in terms of accuracy and factual consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00603v1">Can LLMs Help Improve Analogical Reasoning For Strategic Decisions? Experimental Evidence from Humans and GPT-4</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      This study investigates whether large language models, specifically GPT4, can match human capabilities in analogical reasoning within strategic decision making contexts. Using a novel experimental design involving source to target matching, we find that GPT4 achieves high recall by retrieving all plausible analogies but suffers from low precision, frequently applying incorrect analogies based on superficial similarities. In contrast, human participants exhibit high precision but low recall, selecting fewer analogies yet with stronger causal alignment. These findings advance theory by identifying matching, the evaluative phase of analogical reasoning, as a distinct step that requires accurate causal mapping beyond simple retrieval. While current LLMs are proficient in generating candidate analogies, humans maintain a comparative advantage in recognizing deep structural similarities across domains. Error analysis reveals that AI errors arise from surface level matching, whereas human errors stem from misinterpretations of causal structure. Taken together, the results suggest a productive division of labor in AI assisted organizational decision making where LLMs may serve as broad analogy generators, while humans act as critical evaluators, applying the most contextually appropriate analogies to strategic problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20984v3">UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at https://github.com/tongwu17/SemEval-2025-Task1-UoR-NCL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00557v1">Triggering Hallucinations in LLMs: A Quantitative Study of Prompt-Induced Hallucination in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) present a growing challenge across real-world applications, from healthcare to law, where factual reliability is essential. Despite advances in alignment and instruction tuning, LLMs can still generate outputs that are fluent yet fundamentally untrue. Understanding the cognitive dynamics that underlie these hallucinations remains an open problem. In this study, we propose a prompt-based framework to systematically trigger and quantify hallucination: a Hallucination-Inducing Prompt (HIP), which synthetically fuses semantically distant concepts (e.g., periodic table of elements and tarot divination) in a misleading way, and a Hallucination Quantifying Prompt (HQP), which scores the plausibility, confidence, and coherence of the output. Controlled experiments across multiple LLMs revealed that HIPs consistently produced less coherent and more hallucinated responses than their null-fusion controls. These effects varied across models, with reasoning-oriented LLMs showing distinct profiles from general-purpose ones. Our framework provides a reproducible testbed for studying hallucination vulnerability, and opens the door to developing safer, more introspective LLMs that can detect and self-regulate the onset of conceptual instability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20119v2">Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 8 Pages. This paper has been accepted for presentation at the IEEE Swiss Conference on Data Science (SDS25)
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has advanced significantly in recent years. The complexity of RAG systems, which involve multiple components-such as indexing, retrieval, and generation-along with numerous other parameters, poses substantial challenges for systematic evaluation and quality enhancement. Previous research highlights that evaluating RAG systems is essential for documenting advancements, comparing configurations, and identifying effective approaches for domain-specific applications. This study systematically reviews 63 academic articles to provide a comprehensive overview of state-of-the-art RAG evaluation methodologies, focusing on four key areas: datasets, retrievers, indexing and databases, and the generator component. We observe the feasibility of an automated evaluation approach for each component of a RAG system, leveraging an LLM capable of both generating evaluation datasets and conducting evaluations. In addition, we found that further practical research is essential to provide companies with clear guidance on the do's and don'ts of implementing and evaluating RAG systems. By synthesizing evaluation approaches for key RAG components and emphasizing the creation and adaptation of domain-specific datasets for benchmarking, we contribute to the advancement of systematic evaluation methods and the improvement of evaluation rigor for RAG systems. Furthermore, by examining the interplay between automated approaches leveraging LLMs and human judgment, we contribute to the ongoing discourse on balancing automation and human input, clarifying their respective contributions, limitations, and challenges in achieving robust and reliable evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.08532v3">EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 International Conference on Learning Representations (ICLR) 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various tasks, but they rely on carefully crafted prompts that often demand substantial human effort. To automate this process, in this paper, we propose a novel framework for discrete prompt optimization, called EvoPrompt, which borrows the idea of evolutionary algorithms (EAs) as they exhibit good performance and fast convergence. To enable EAs to work on discrete prompts, which are natural language expressions that need to be coherent and human-readable, we connect LLMs with EAs. This approach allows us to simultaneously leverage the powerful language processing capabilities of LLMs and the efficient optimization performance of EAs. Specifically, abstaining from any gradients or parameters, EvoPrompt starts from a population of prompts and iteratively generates new prompts with LLMs based on the evolutionary operators, improving the population based on the development set. We optimize prompts for both closed- and open-source LLMs including GPT-3.5 and Alpaca, on 31 datasets covering language understanding, generation tasks, as well as BIG-Bench Hard (BBH) tasks. EvoPrompt significantly outperforms human-engineered prompts and existing methods for automatic prompt generation (e.g., up to 25% on BBH). Furthermore, EvoPrompt demonstrates that connecting LLMs with EAs creates synergies, which could inspire further research on the combination of LLMs and conventional algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21036v2">Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks?</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 accepted by DBSec25
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03621v3">Network-aided Efficient LLM Services With Denoising-inspired Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.18010
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, leading to their increasing adoption in diverse services delivered through wireless networks. There is a growing trend toward longer prompts to better leverage LLMs' capabilities and address difficult tasks. However, longer prompts not only increase data transmission costs but also require more computing resources and processing time, which impacts overall system efficiency and user experience. To address this challenge, we propose Joint Power and Prompt Optimization (JPPO), a framework that combines Small Language Model (SLM)-based prompt compression with wireless power allocation optimization. By deploying SLM at edge devices for prompt compression and employing Deep Reinforcement Learning (DRL) for joint optimization of compression ratio and transmission power, JPPO effectively balances service quality with resource efficiency. Furthermore, inspired by denoising diffusion models, we design a denoising-inspired prompt compression approach that iteratively compresses prompts by gradually removing non-critical information, further enhancing the framework's performance. Experimental results with long prompt tokens demonstrate that our framework achieves high service fidelity while optimizing power usage in wireless LLM services, significantly reducing the total service response time. With our DRL-based JPPO, the framework maintains fidelity comparable to the no-compression baseline while still achieving a 17% service time reduction through adaptive compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00368v1">Urban Air Mobility as a System of Systems: An LLM-Enhanced Holonic Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Urban Air Mobility (UAM) is an emerging System of System (SoS) that faces challenges in system architecture, planning, task management, and execution. Traditional architectural approaches struggle with scalability, adaptability, and seamless resource integration within dynamic and complex environments. This paper presents an intelligent holonic architecture that incorporates Large Language Model (LLM) to manage the complexities of UAM. Holons function semi autonomously, allowing for real time coordination among air taxis, ground transport, and vertiports. LLMs process natural language inputs, generate adaptive plans, and manage disruptions such as weather changes or airspace closures.Through a case study of multimodal transportation with electric scooters and air taxis, we demonstrate how this architecture enables dynamic resource allocation, real time replanning, and autonomous adaptation without centralized control, creating more resilient and efficient urban transportation networks. By advancing decentralized control and AI driven adaptability, this work lays the groundwork for resilient, human centric UAM ecosystems, with future efforts targeting hybrid AI integration and real world validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00342v1">LLMPrism: Black-box Performance Diagnosis for Production LLM Training Platforms</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have brought about revolutionary changes in diverse fields, rendering LLM training of utmost importance for modern enterprises. To meet this demand, multi-tenant large-scale LLM training platforms have been built to offer LLM training services. Nevertheless, due to the complexity and synchronous nature of LLM training process, performance issues occur frequently and can result in substantial resource wastage. The limited visibility from the perspective of platform providers impedes existing profiling methods and poses challenges to the monitoring and diagnosis of the performance of LLM training jobs. For the first time, this paper proposes the utilization of underlying network flow data to reconstruct the training timelines of jobs based on the distinct characteristics in the LLM training procedure. We design LLMPrism, the first black-box performance diagnosis system for LLM training platforms. By progressively recognizing LLM training jobs, identifying their parallelism strategies, and reconstructing the training timelines, LLMPrism achieves non-intrusive, lightweight, and continuous monitoring of LLM training systems. Leveraging this monitoring capability, it further effectively diagnoses potential performance issues. Since Oct. 2024, LLMPrism has been deployed on our large-scale production Platform-X, in which the evaluations and deployment experiences demonstrate that LLMPrism can achieve accurate timeline reconstruction with an error within 0.3% and effectively diagnose various performance issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15546v2">A Framework for Testing and Adapting REST APIs as LLM Tools</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16473v2">TerEffic: Highly Efficient Ternary LLM Inference on FPGA</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) efficiently on edge devices is often constrained by limited memory capacity and high power consumption. Low-bit quantization methods, particularly ternary quantization, have demonstrated significant potential in preserving model accuracy while substantially decreasing memory footprint and computational costs. However, existing general-purpose architectures and accelerators have not fully exploited the advantages of low-bit quantization due to insufficient specialized hardware support. We introduce TerEffic, an FPGA-based architecture tailored for ternary-quantized LLM inference. The proposed system offers flexibility through reconfigurable hardware to meet various system requirements. We evaluated two representative configurations: a fully on-chip design that stores all weights within on-chip memories, scaling out using multiple FPGAs, and an HBM-assisted design capable of accommodating larger models on a single FPGA board. Experimental results demonstrate significant performance and energy efficiency improvements. For single-batch inference on a 370 M-parameter model, our fully on-chip architecture achieves 16,300 tokens/second, delivering a throughput 192 times higher than NVIDIA Jetson Orin Nano with a power efficiency of 455 tokens/second/W, marking a 19-fold improvement. The HBM-assisted architecture processes 727 tokens/second for a larger 2.7B-parameter model, which is 3 times of the throughput of NVIDIA A100, while consuming only 46W, resulting in a power efficiency of 16 tokens/second/W, an 8-fold improvement over the A100.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04532v3">QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 The first three authors contribute equally to this project and are listed in the alphabetical order. Yujun Lin leads the quantization algorithm, Haotian Tang and Shang Yang lead the GPU kernels and the serving system. Code is available at https://github.com/mit-han-lab/omniserve
    </div>
    <details class="paper-abstract">
      Quantization can accelerate large language model (LLM) inference. Going beyond INT8 quantization, the research community is actively exploring even lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization techniques only accelerate low-batch, edge LLM inference, failing to deliver performance gains in large-batch, cloud-based LLM serving. We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs. To address this challenge, we introduce QoQ, a W4A8KV4 quantization algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands for quattuor-octo-quattuor, which represents 4-8-4 in Latin. QoQ is implemented by the QServe inference library that achieves measured speedup. The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by operations on low-throughput CUDA cores. Building upon this insight, in QoQ algorithm, we introduce progressive quantization that can allow low dequantization overhead in W4A8 GEMM. Additionally, we develop SmoothAttention to effectively mitigate the accuracy degradation incurred by 4-bit KV quantization. In the QServe system, we perform compute-aware weight reordering and take advantage of register-level parallelism to reduce dequantization latency. We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization. As a result, QServe improves the maximum achievable serving throughput of Llama-3-8B by 1.2x on A100, 1.4x on L40S; and Qwen1.5-72B by 2.4x on A100, 3.5x on L40S, compared to TensorRT-LLM. Remarkably, QServe on L40S GPU can achieve even higher throughput than TensorRT-LLM on A100. Thus, QServe effectively reduces the dollar cost of LLM serving by 3x. Code is available at https://github.com/mit-han-lab/omniserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00240v1">LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The increasing complexity and scale of the Internet of Things (IoT) have made security a critical concern. This paper presents a novel Large Language Model (LLM)-based framework for comprehensive threat detection and prevention in IoT environments. The system integrates lightweight LLMs fine-tuned on IoT-specific datasets (IoT-23, TON_IoT) for real-time anomaly detection and automated, context-aware mitigation strategies optimized for resource-constrained devices. A modular Docker-based deployment enables scalable and reproducible evaluation across diverse network conditions. Experimental results in simulated IoT environments demonstrate significant improvements in detection accuracy, response latency, and resource efficiency over traditional security methods. The proposed framework highlights the potential of LLM-driven, autonomous security solutions for future IoT ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08498v5">"Reasoning" with Rhetoric: On the Style-Evidence Tradeoff in LLM-Generated Counter-Arguments</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 22 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) play a key role in generating evidence-based and stylistic counter-arguments, yet their effectiveness in real-world applications has been underexplored. Previous research often neglects the balance between evidentiality and style, which are crucial for persuasive arguments. To address this, we evaluated the effectiveness of stylized evidence-based counter-argument generation in Counterfire, a new dataset of 38,000 counter-arguments generated by revising counter-arguments to Reddit's ChangeMyView community to follow different discursive styles. We evaluated generic and stylized counter-arguments from basic and fine-tuned models such as GPT-3.5, PaLM-2, and Koala-13B, as well as newer models (GPT-4o, Claude Haiku, LLaMA-3.1) focusing on rhetorical quality and persuasiveness. Our findings reveal that humans prefer stylized counter-arguments over the original outputs, with GPT-3.5 Turbo performing well, though still not reaching human standards of rhetorical quality nor persuasiveness. Additionally, our work created a novel argument triplets dataset for studying style control, with human preference labels that provide insights into the tradeoffs between evidence integration and argument quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00234v1">Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Many methods for improving Large Language Model (LLM) agents for sequential decision-making tasks depend on task-specific knowledge engineering--such as prompt tuning, curated in-context examples, or customized observation and action spaces. Using these approaches, agent performance improves with the quality or amount of knowledge engineering invested. Instead, we investigate how LLM agents can automatically improve their performance by learning in-context from their own successful experiences on similar tasks. Rather than relying on task-specific knowledge engineering, we focus on constructing and refining a database of self-generated examples. We demonstrate that even a naive accumulation of successful trajectories across training tasks boosts test performance on three benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%)--matching the performance the initial agent achieves if allowed two to three attempts per task. We then introduce two extensions: (1) database-level selection through population-based training to identify high-performing example collections, and (2) exemplar-level selection that retains individual trajectories based on their empirical utility as in-context examples. These extensions further enhance performance, achieving 91% on ALFWorld--matching more complex approaches that employ task-specific components and prompts. Our results demonstrate that automatic trajectory database construction offers a compelling alternative to labor-intensive knowledge engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07963v2">Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 22 pages, 12 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
    </details>
</div>
