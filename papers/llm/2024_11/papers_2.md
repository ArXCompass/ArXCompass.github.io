# llm - 2024_11

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16116v1">LLM Augmentations to support Analytical Reasoning over Multiple Documents</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 2024 IEEE International Conference on Big Data (IEEE BigData 2024)
    </div>
    <details class="paper-abstract">
      Building on their demonstrated ability to perform a variety of tasks, we investigate the application of large language models (LLMs) to enhance in-depth analytical reasoning within the context of intelligence analysis. Intelligence analysts typically work with massive dossiers to draw connections between seemingly unrelated entities, and uncover adversaries' plans and motives. We explore if and how LLMs can be helpful to analysts for this task and develop an architecture to augment the capabilities of an LLM with a memory module called dynamic evidence trees (DETs) to develop and track multiple investigation threads. Through extensive experiments on multiple datasets, we highlight how LLMs, as-is, are still inadequate to support intelligence analysts and offer recommendations to improve LLMs for such intricate reasoning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16111v1">LLMPirate: LLMs for Black-box Hardware IP Piracy</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 Accepted by NDSS Symposium 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled the ability to effectively analyze and generate code nearly instantaneously, resulting in their widespread adoption in software development. Following this advancement, researchers and companies have begun integrating LLMs across the hardware design and verification process. However, these highly potent LLMs can also induce new attack scenarios upon security vulnerabilities across the hardware development process. One such attack vector that has not been explored is intellectual property (IP) piracy. Given that this attack can manifest as rewriting hardware designs to evade piracy detection, it is essential to thoroughly evaluate LLM capabilities in performing this task and assess the mitigation abilities of current IP piracy detection tools. Therefore, in this work, we propose LLMPirate, the first LLM-based technique able to generate pirated variations of circuit designs that successfully evade detection across multiple state-of-the-art piracy detection tools. We devise three solutions to overcome challenges related to integration of LLMs for hardware circuit designs, scalability to large circuits, and effectiveness, resulting in an end-to-end automated, efficient, and practical formulation. We perform an extensive experimental evaluation of LLMPirate using eight LLMs of varying sizes and capabilities and assess their performance in pirating various circuit designs against four state-of-the-art, widely-used piracy detection tools. Our experiments demonstrate that LLMPirate is able to consistently evade detection on 100% of tested circuits across every detection tool. Additionally, we showcase the ramifications of LLMPirate using case studies on IBEX and MOR1KX processors and a GPS module, that we successfully pirate. We envision that our work motivates and fosters the development of better IP piracy detection tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16020v1">TransCompressor: LLM-Powered Multimodal Data Compression for Smart Transportation</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The incorporation of Large Language Models (LLMs) into smart transportation systems has paved the way for improving data management and operational efficiency. This study introduces TransCompressor, a novel framework that leverages LLMs for efficient compression and decompression of multimodal transportation sensor data. TransCompressor has undergone thorough evaluation with diverse sensor data types, including barometer, speed, and altitude measurements, across various transportation modes like buses, taxis, and MTRs. Comprehensive evaluation illustrates the effectiveness of TransCompressor in reconstructing transportation sensor data at different compression ratios. The results highlight that, with well-crafted prompts, LLMs can utilize their vast knowledge base to contribute to data compression processes, enhancing data storage, analysis, and retrieval in smart transportation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15998v1">PIANIST: Learning Partially Observable World Models with LLMs for Multi-Agent Decision Making</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 Published at Language Gamification Workshop 2024 @ NeurIPS
    </div>
    <details class="paper-abstract">
      Effective extraction of the world knowledge in LLMs for complex decision-making tasks remains a challenge. We propose a framework PIANIST for decomposing the world model into seven intuitive components conducive to zero-shot LLM generation. Given only the natural language description of the game and how input observations are formatted, our method can generate a working world model for fast and efficient MCTS simulation. We show that our method works well on two different games that challenge the planning and decision making skills of the agent for both language and non-language based action taking, without any training on domain-specific training data or explicitly defined world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15997v1">Ensuring Fair LLM Serving Amid Diverse Applications</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      In a multi-tenant large language model (LLM) serving platform hosting diverse applications, some users may submit an excessive number of requests, causing the service to become unavailable to other users and creating unfairness. Existing fairness approaches do not account for variations in token lengths across applications and multiple LLM calls, making them unsuitable for such platforms. To address the fairness challenge, this paper analyzes millions of requests from thousands of users on MS CoPilot, a real-world multi-tenant LLM platform hosted by Microsoft. Our analysis confirms the inadequacy of existing methods and guides the development of FairServe, a system that ensures fair LLM access across diverse applications. FairServe proposes application-characteristic aware request throttling coupled with a weighted service counter based scheduling technique to curb abusive behavior and ensure fairness. Our experimental results on real-world traces demonstrate FairServe's superior performance compared to the state-of-the-art method in ensuring fairness. We are actively working on deploying our system in production, expecting to benefit millions of customers world-wide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13352v3">AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 Updated version after fixing a bug in the Llama implementation and updating the travel suite
    </div>
    <details class="paper-abstract">
      AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner.. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15982v1">Anda: Unlocking Efficient LLM Inference with a Variable-Length Grouped Activation Data Format</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 To appear in 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA 2025)
    </div>
    <details class="paper-abstract">
      The widely-used, weight-only quantized large language models (LLMs), which leverage low-bit integer (INT) weights and retain floating-point (FP) activations, reduce storage requirements while maintaining accuracy. However, this shifts the energy and latency bottlenecks towards the FP activations that are associated with costly memory accesses and computations. Existing LLM accelerators focus primarily on computation optimizations, overlooking the potential of jointly optimizing FP computations and data movement, particularly for the dominant FP-INT GeMM operations in LLM inference. To address these challenges, we investigate the sensitivity of activation precision across various LLM modules and its impact on overall model accuracy. Based on our findings, we first propose the Anda data type: an adaptive data format with group-shared exponent bits and dynamic mantissa bit allocation. Secondly, we develop an iterative post-training adaptive precision search algorithm that optimizes the bit-width for different LLM modules to balance model accuracy, energy efficiency, and inference speed. Lastly, a suite of hardware optimization techniques is proposed to maximally exploit the benefits of the Anda format. These include a bit-plane-based data organization scheme, Anda-enhanced processing units with bit-serial computation, and a runtime bit-plane Anda compressor to simultaneously optimize storage, computation, and memory footprints. Our evaluations on FPINT GeMM operations show that Anda achieves a 2.4x speedup, 4.0x area efficiency, and 3.1x energy efficiency improvement on average for popular LLMs including OPT, LLaMA, and LLaMA-2 series over the GPU-like FP-FP baseline. Anda demonstrates strong adaptability across various application scenarios, accuracy requirements, and system performance, enabling efficient LLM inference across a wide range of deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10313v2">How Far Are We From AGI: Are LLMs All We Need?</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      The evolution of artificial intelligence (AI) has profoundly impacted human society, driving significant advancements in multiple sectors. AGI, distinguished by its ability to execute diverse real-world tasks with efficiency and effectiveness comparable to human intelligence, reflects a paramount milestone in AI evolution. While existing studies have reviewed specific advancements in AI and proposed potential paths to AGI, such as large language models (LLMs), they fall short of providing a thorough exploration of AGI's definitions, objectives, and developmental trajectories. Unlike previous survey papers, this work goes beyond summarizing LLMs by addressing key questions about our progress toward AGI and outlining the strategies essential for its realization through comprehensive analysis, in-depth discussions, and novel insights. We start by articulating the requisite capability frameworks for AGI, integrating the internal, interface, and system dimensions. As the realization of AGI requires more advanced capabilities and adherence to stringent constraints, we further discuss necessary AGI alignment technologies to harmonize these factors. Notably, we emphasize the importance of approaching AGI responsibly by first defining the key levels of AGI progression, followed by the evaluation framework that situates the status quo, and finally giving our roadmap of how to reach the pinnacle of AGI. Moreover, to give tangible insights into the ubiquitous impact of the integration of AI, we outline existing challenges and potential pathways toward AGI in multiple domains. In sum, serving as a pioneering exploration into the current state and future trajectory of AGI, this paper aims to foster a collective comprehension and catalyze broader public discussions among researchers and practitioners on AGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07921v3">AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 Published as a conference paper at COLM 2024 (https://colmweb.org/index.html)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17741v1">Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      The widespread adoption of LLMs has driven an exponential rise in their deployment, imposing substantial demands on inference clusters. These clusters must handle numerous concurrent queries for different LLM downstream tasks. To handle multi-task settings with vast LLM parameter counts, methods like Low-Rank Adaptation (LoRA) enable task-specific fine-tuning while sharing most of the base LLM model across tasks. Hence, they allow concurrent task serving with minimal memory requirements. However, existing LLM serving systems face inefficiencies: they overlook workload heterogeneity, impose high link bandwidth from frequent adapter loading, and suffer from head-of-line blocking in their schedulers. To address these challenges, we present Chameleon, a novel LLM serving system optimized for many adapter environments, that relies on two core ideas: adapter caching and adapter-aware scheduling. First, Chameleon caches popular adapters in GPU memory, minimizing the adapter loading times. Importantly, it uses the otherwise idle GPU memory, avoiding extra memory costs. Second, Chameleon uses a non-preemptive multi-queue scheduling to efficiently account for workload heterogeneity. In this way, Chameleon simultaneously prevents head of line blocking and starvation. We implement Chameleon on top of a state-of-the-art LLM serving platform and evaluate it with real-world production traces and open-source LLMs. Under high loads, Chameleon reduces P99 and P50 TTFT latency by 80.7% and 48.1%, respectively, while improving throughput by 1.5x compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15898v1">Towards the LLM-Based Generation of Formal Specifications from Natural-Language Contracts: Early Experiments with Symboleo</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 9 pages, 1 figure, 2 tables, submitted to a workshop
    </div>
    <details class="paper-abstract">
      Over the past decade, different domain-specific languages (DSLs) were proposed to formally specify requirements stated in legal contracts, mainly for analysis but also for code generation. Symboleo is a promising language in that area. However, writing formal specifications from natural-language contracts is a complex task, especial for legal experts who do not have formal language expertise. This paper reports on an exploratory experiment targeting the automated generation of Symboleo specifications from business contracts in English using Large Language Models (LLMs). Combinations (38) of prompt components are investigated (with/without the grammar, semantics explanations, 0 to 3 examples, and emotional prompts), mainly on GPT-4o but also to a lesser extent on 4 other LLMs. The generated specifications are manually assessed against 16 error types grouped into 3 severity levels. Early results on all LLMs show promising outcomes (even for a little-known DSL) that will likely accelerate the specification of legal contracts. However, several observed issues, especially around grammar/syntax adherence and environment variable identification (49%), suggest many areas where potential improvements should be investigated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12830v3">Incorporating Metabolic Information into LLMs for Anomaly Detection in Clinical Time-Series</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Anomaly detection in clinical time-series holds significant potential in identifying suspicious patterns in different biological parameters. In this paper, we propose a targeted method that incorporates the clinical domain knowledge into LLMs to improve their ability to detect anomalies. We introduce the Metabolism Pathway-driven Prompting (MPP) method, which integrates the information about metabolic pathways to better capture the structural and temporal changes in biological samples. We applied our method for doping detection in sports, focusing on steroid metabolism, and evaluated using real-world data from athletes. The results show that our method improves anomaly detection performance by leveraging metabolic context, providing a more nuanced and accurate prediction of suspicious samples in athletes' profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15871v1">Hiding Communication Cost in Distributed LLM Training via Micro-batch Co-execution</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      The growth of Large Language Models (LLMs) has necessitated large-scale distributed training. Highly optimized frameworks, however, still suffer significant losses in Model FLOPS utilization (often below 50%) due to large communication volumes. Meanwhile, our comprehensive profiling shows that the computation- and communication-intensive operators overlap well. This paper introduces DHelix, a novel micro-structure that dramatically improves the efficiency of LLM training inspired by the DNA structure. Central to DHelix's design is Strand Interleaving (SI), which views the continuous stream of training micro-batches through a GPU as two strands. DHelix juxtaposes the forward and backward passes of the two strands and performs a systematic optimization for an SI plan that co-schedules the operators from the opposite strands, enabled by operator-level overlap profiling results and a dynamic-programming based search algorithm. Meanwhile, DHelix enables the two strands to share model states and space for activation data, effectively accommodating two micro-batches with under 3% extra memory space. Dhelix seamlessly integrates with all forms of existing data/model parallelism, the most challenging being pipeline parallelism, thanks to its unique model folding design that results in a W-shaped pipeline. We evaluate DHelix training with the popular Llama and GPT dense models, plus the Phi Mixture of Expert (MoE) model, across 3 GPU clusters (A40, A800, and H100). Results show that it achieves 12-40% (up to 58% MFU) and 2-29% (up to 71% MFU) improvement on the 64-A40 and 64-A800 clusters, respectively, significantly outperforming state-of-the-art methods. On the H100 cluster, though the faster network reduces DHelix's profit margin, it makes cross-node tensor parallelism promising, a practice currently prohibitive due to communication costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00053v1">LeMoLE: LLM-Enhanced Mixture of Linear Experts for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Recent research has shown that large language models (LLMs) can be effectively used for real-world time series forecasting due to their strong natural language understanding capabilities. However, aligning time series into semantic spaces of LLMs comes with high computational costs and inference complexity, particularly for long-range time series generation. Building on recent advancements in using linear models for time series, this paper introduces an LLM-enhanced mixture of linear experts for precise and efficient time series forecasting. This approach involves developing a mixture of linear experts with multiple lookback lengths and a new multimodal fusion mechanism. The use of a mixture of linear experts is efficient due to its simplicity, while the multimodal fusion mechanism adaptively combines multiple linear experts based on the learned features of the text modality from pre-trained large language models. In experiments, we rethink the need to align time series to LLMs by existing time-series large language models and further discuss their efficiency and effectiveness in time series forecasting. Our experimental results show that the proposed LeMoLE model presents lower prediction errors and higher computational efficiency than existing LLM models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15764v1">LLM Online Spatial-temporal Signal Reconstruction Under Noise</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      This work introduces the LLM Online Spatial-temporal Reconstruction (LLM-OSR) framework, which integrates Graph Signal Processing (GSP) and Large Language Models (LLMs) for online spatial-temporal signal reconstruction. The LLM-OSR utilizes a GSP-based spatial-temporal signal handler to enhance graph signals and employs LLMs to predict missing values based on spatiotemporal patterns. The performance of LLM-OSR is evaluated on traffic and meteorological datasets under varying Gaussian noise levels. Experimental results demonstrate that utilizing GPT-4-o mini within the LLM-OSR is accurate and robust under Gaussian noise conditions. The limitations are discussed along with future research insights, emphasizing the potential of combining GSP techniques with LLMs for solving spatiotemporal prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15692v1">DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have opened new avenues for accelerating drug discovery processes. Despite their potential, several critical challenges remain unsolved, particularly in translating theoretical ideas into practical applications within the highly specialized field of pharmaceutical research, limiting practitioners from leveraging the latest AI development in drug discovery. To this end, we introduce DrugAgent, a multi-agent framework aimed at automating machine learning (ML) programming in drug discovery. DrugAgent incorporates domain expertise by identifying specific requirements and building domain-specific tools, while systematically exploring different ideas to find effective solutions. A preliminary case study demonstrates DrugAgent's potential to overcome key limitations LLMs face in drug discovery, moving toward AI-driven innovation. For example, DrugAgent is able to complete the ML programming pipeline end-to-end, from data acquisition to performance evaluation for the ADMET prediction task, and finally select the best model, where the random forest model achieves an F1 score of 0.92 when predicting absorption using the PAMPA dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14629v3">Can LLMs Learn by Teaching for Better Reasoning? A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Teaching to improve student models (e.g., knowledge distillation) is an extensively studied methodology in LLMs. However, for humans, teaching improves not only students but also teachers, by fostering more rigorous and clear reasoning as well as knowledge building. We ask: Can LLMs also learn by teaching (LbT) for better reasoning? If the answer is yes, we can potentially unlock the possibility of continuously advancing the models without solely relying on human-produced data or stronger models. In this paper, we provide a preliminary exploration on this question. We show that LbT ideas can be incorporated into existing LLM training/prompting pipelines and bring improvements. Specifically, we design three methods, each mimicking one of the three levels of LbT: observing students' feedback, learning from the feedback, and learning iteratively, with the goals of improving answer accuracy without training or improving models' inherent capability with fine-tuning. We reveal some findings: (1) Teaching materials that make it easier for students to learn have clearer and more accurate logic when using in-context learning as the student's "learning" method; (2) Weak-to-strong generalization: LbT might help improve strong models by teaching weak models; (3) Diversity in students might help: teaching multiple students could be better than teaching one student or the teacher itself. We hope that our exploration can inspire future research on LbT and more broadly adopting the advanced techniques in education to improve LLMs. The code and website are at https://github.com/imagination-research/lbt and https://sites.google.com/view/llm-learning-by-teaching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16003v1">eFedLLM: Efficient LLM Inference Based on Federated Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) herald a transformative era in artificial intelligence (AI). However, the expansive scale of data and parameters of LLMs requires high-demand computational and memory resources, restricting their accessibility to a broader range of users and researchers. This paper introduces an effective approach that enhances the operational efficiency and affordability of LLM inference. By utilizing transformer-based federated learning (FL) with model-parallel distributed training, our model efficiently distributes the computational loads and memory requirements across a network of participants. This strategy permits users, especially those with limited resources to train state-of-the-art LLMs collaboratively. We also innovate an incentive mechanism within the FL framework, rewarding constructive contributions and filtering out malicious activities, thereby safeguarding the integrity and reliability of the training process. Concurrently, we leverage memory hierarchy strategies and Singular Value Decomposition (SVD) on weight matrices to boost computational and memory efficiencies further. Our results, derived from formulaic analyses and numerical calculations, demonstrate significant optimization of resource use and democratize access to cutting-edge LLMs, ensuring that a wide scale of users can both contribute to and benefit from these advanced models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15664v1">Enabling Efficient Serverless Inference Serving for LLM (Large Language Model) in the Cloud</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 12 pages, 7 figures, TUM Cloud Computing Seminar
    </div>
    <details class="paper-abstract">
      This review report discusses the cold start latency in serverless inference and existing solutions. It particularly reviews the ServerlessLLM method, a system designed to address the cold start problem in serverless inference for large language models. Traditional serverless approaches struggle with high latency due to the size of LLM checkpoints and the overhead of initializing GPU resources. ServerlessLLM introduces a multitier checkpoint loading system, leveraging underutilized GPU memory and storage to reduce startup times by 6--8x compared to existing methods. It also proposes live inference migration and a startup-time-optimized model scheduler, ensuring efficient resource allocation and minimizing delays. This system significantly improves performance and scalability in serverless environments for LLM workloads. Besides ServerlessLLM, several other methods from recent research literature, including Rainbowcake, are reviewed in this paper. Further discussions explore how FaaS providers tackle cold starts and the possible future scopes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15662v1">Gaps Between Research and Practice When Measuring Representational Harms Caused by LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 NeurIPS 2024 Workshop on Evaluating Evaluations (EvalEval)
    </div>
    <details class="paper-abstract">
      To facilitate the measurement of representational harms caused by large language model (LLM)-based systems, the NLP research community has produced and made publicly available numerous measurement instruments, including tools, datasets, metrics, benchmarks, annotation instructions, and other techniques. However, the research community lacks clarity about whether and to what extent these instruments meet the needs of practitioners tasked with developing and deploying LLM-based systems in the real world, and how these instruments could be improved. Via a series of semi-structured interviews with practitioners in a variety of roles in different organizations, we identify four types of challenges that prevent practitioners from effectively using publicly available instruments for measuring representational harms caused by LLM-based systems: (1) challenges related to using publicly available measurement instruments; (2) challenges related to doing measurement in practice; (3) challenges arising from measurement tasks involving LLM-based systems; and (4) challenges specific to measuring representational harms. Our goal is to advance the development of instruments for measuring representational harms that are well-suited to practitioner needs, thus better facilitating the responsible development and deployment of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09688v2">Squeezed Attention: Accelerating Long Context Length LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Emerging Large Language Model (LLM) applications require long input prompts to perform complex downstream tasks like document analysis and code generation. For these long context length applications, the length of the input prompt poses a significant challenge in terms of inference efficiency since the inference costs increase linearly with sequence length. However, for many of these applications, much of the context in the prompt is fixed across different user inputs, thereby providing the opportunity to perform offline optimizations to process user inputs quickly, as they are received. In this work, we propose Squeezed Attention as a mechanism to accelerate LLM applications where a large portion of the input prompt is fixed. We first leverage K-means clustering offline to group the keys for the fixed context based on semantic similarity and represent each cluster with a single centroid value. During inference, we compare query tokens from the user input with the centroids to predict which of the keys from the fixed context are semantically relevant and need to be loaded during inference. We then compute exact attention using only these important keys from the fixed context, thereby reducing bandwidth and computational costs. We also extend our method to use a hierarchical centroid lookup to identify important keys, which can reduce the complexity of attention from linear to logarithmic with respect to the context length. We implement optimized Triton kernels for centroid comparison and sparse FlashAttention with important keys, achieving more than 4x speedups during both the prefill and generation phases for long-context inference. Furthermore, we have extensively evaluated our method on various long-context benchmarks including LongBench, where it achieves a 3x reduction in KV cache budget without accuracy loss and up to an 8x reduction with <0.5 point accuracy gap for various models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15526v2">SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a clear trend towards language models with an ever-increasing number of parameters, as well as the growing training overhead and memory usage. Distributed training, particularly through Sharded Data Parallelism (ShardedDP) which partitions optimizer states among workers, has emerged as a crucial technique to mitigate training time and memory usage. Yet, a major challenge in the scalability of ShardedDP is the intensive communication of weights and gradients. While compression techniques can alleviate this issue, they often result in worse accuracy. Driven by this limitation, we propose SDP4Bit (Toward 4Bit Communication Quantization in Sharded Data Parallelism for LLM Training), which effectively reduces the communication of weights and gradients to nearly 4 bits via two novel techniques: quantization on weight differences, and two-level gradient smooth quantization. Furthermore, SDP4Bit presents an algorithm-system co-design with runtime optimization to minimize the computation overhead of compression. In addition to the theoretical guarantees of convergence, we empirically evaluate the accuracy of SDP4Bit on the pre-training of GPT models with up to 6.7 billion parameters, and the results demonstrate a negligible impact on training loss. Furthermore, speed experiments show that SDP4Bit achieves up to 4.08$\times$ speedup in end-to-end throughput on a scale of 128 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15269v3">Aligning LLM Agents by Learning Latent Preference from User Edits</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      We study interactive learning of LLM-based language agents based on user edits made to the agent's output. In a typical setting such as writing assistants, the user interacts with a language agent to generate a response given a context, and may optionally edit the agent response to personalize it based on their latent preference, in addition to improving the correctness. The edit feedback is naturally generated, making it a suitable candidate for improving the agent's alignment with the user's preference, and for reducing the cost of user edits over time. We propose a learning framework, PRELUDE that infers a description of the user's latent preference based on historic edit data. The inferred user preference descriptions are used to define prompts for generating responses in the future. This avoids fine-tuning the agent, which is costly, challenging to scale with the number of users, and may even degrade its performance on other tasks. Furthermore, learning descriptive preference improves interpretability, allowing the user to view and modify the learned preference. However, user preference can be complex, subtle, and vary based on context, making it challenging to learn. To address this, we propose a simple yet effective algorithm named CIPHER that leverages the LLM to infer the user preference for a given context based on user edits. In the future, CIPHER retrieves inferred preferences from the k-closest contexts in the history, and forms an aggregate preference for response generation. We introduce two interactive environments -- summarization and email writing, and use a GPT-4 simulated user for evaluation. On both tasks, CIPHER outperforms several baselines by achieving the lowest edit distance cost while only having a small overhead in LLM query cost. Our analysis reports that user preferences learned by CIPHER show significant similarity to the ground truth latent preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15558v1">Reassessing Layer Pruning in LLMs: New Insights and Methods</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have achieved remarkable success across various domains, their considerable scale necessitates substantial computational resources, posing significant challenges for deployment in resource-constrained environments. Layer pruning, as a simple yet effective compression method, removes layers of a model directly, reducing computational overhead. However, what are the best practices for layer pruning in LLMs? Are sophisticated layer selection metrics truly effective? Does the LoRA (Low-Rank Approximation) family, widely regarded as a leading method for pruned model fine-tuning, truly meet expectations when applied to post-pruning fine-tuning? To answer these questions, we dedicate thousands of GPU hours to benchmarking layer pruning in LLMs and gaining insights across multiple dimensions. Our results demonstrate that a simple approach, i.e., pruning the final 25\% of layers followed by fine-tuning the \texttt{lm\_head} and the remaining last three layer, yields remarkably strong performance. Following this guide, we prune Llama-3.1-8B-It and obtain a model that outperforms many popular LLMs of similar size, such as ChatGLM2-6B, Vicuna-7B-v1.5, Qwen1.5-7B and Baichuan2-7B. We release the optimal model weights on Huggingface, and the code is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16736v1">ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      The advancement and extensive application of large language models (LLMs) have been remarkable, including their use in scientific research assistance. However, these models often generate scientifically incorrect or unsafe responses, and in some cases, they may encourage users to engage in dangerous behavior. To address this issue in the field of chemistry, we introduce ChemSafetyBench, a benchmark designed to evaluate the accuracy and safety of LLM responses. ChemSafetyBench encompasses three key tasks: querying chemical properties, assessing the legality of chemical uses, and describing synthesis methods, each requiring increasingly deeper chemical knowledge. Our dataset has more than 30K samples across various chemical materials. We incorporate handcrafted templates and advanced jailbreaking scenarios to enhance task diversity. Our automated evaluation framework thoroughly assesses the safety, accuracy, and appropriateness of LLM responses. Extensive experiments with state-of-the-art LLMs reveal notable strengths and critical vulnerabilities, underscoring the need for robust safety measures. ChemSafetyBench aims to be a pivotal tool in developing safer AI technologies in chemistry. Our code and dataset are available at https://github.com/HaochenZhao/SafeAgent4Chem. Warning: this paper contains discussions on the synthesis of controlled chemicals using AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15501v1">Instruct or Interact? Exploring and Eliciting LLMs' Capability in Code Snippet Adaptation Through Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 12 pages, 10 figures, accepted by ICSE 2025
    </div>
    <details class="paper-abstract">
      Code snippet adaptation is a fundamental activity in the software development process. Unlike code generation, code snippet adaptation is not a "free creation", which requires developers to tailor a given code snippet in order to fit specific requirements and the code context. Recently, large language models (LLMs) have confirmed their effectiveness in the code generation task with promising results. However, their performance on adaptation, a reuse-oriented and context-dependent code change prediction task, is still unclear. To bridge this gap, we conduct an empirical study to investigate the performance and issues of LLMs on the adaptation task. We first evaluate the adaptation performances of three popular LLMs and compare them to the code generation task. Our result indicates that their adaptation ability is weaker than generation, with a nearly 15% decrease on pass@1 and more context-related errors. By manually inspecting 200 cases, we further investigate the causes of LLMs' sub-optimal performance, which can be classified into three categories, i.e., Unclear Requirement, Requirement Misalignment and Context Misapplication. Based on the above empirical research, we propose an interactive prompting approach to eliciting LLMs' adaptation ability. Experimental result reveals that our approach greatly improve LLMs' adaptation performance. The best-performing Human-LLM interaction successfully solves 159 out of the 202 identified defects and improves the pass@1 and pass@5 by over 40% compared to the initial instruction-based prompt. Considering human efforts, we suggest multi-agent interaction as a trade-off, which can achieve comparable performance with excellent generalization ability. We deem that our approach could provide methodological assistance for autonomous code snippet reuse and adaptation with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15484v1">Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs: A Case Study in Thai</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 ACL-SRW 2024. Our code and dataset are publicly available at https://github.com/parinzee/seed-free-synthetic-instruct
    </div>
    <details class="paper-abstract">
      We present a synthetic data approach for instruction-tuning large language models (LLMs) for low-resource languages in a data-efficient manner, specifically focusing on Thai. We identify three key properties that contribute to the effectiveness of instruction-tuning datasets: fluency, diversity, and cultural context. We propose a seed-data-free framework for generating synthetic instruction-tuning data that incorporates these essential properties. Our framework employs an LLM to generate diverse topics, retrieve relevant contexts from Wikipedia, and create instructions for various tasks, such as question answering, summarization, and conversation. The experimental results show that our best-performing synthetic dataset, which incorporates all three key properties, achieves competitive performance using only 5,000 instructions when compared to state-of-the-art Thai LLMs trained on hundreds of thousands of instructions. Our code and dataset are publicly available at https://github.com/parinzee/seed-free-synthetic-instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15477v1">Towards Robust Evaluation of Unlearning in LLMs via Data Transformations</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 Accepted at EMNLP 2024 Findings; 21 pages (5 page main content + references + appendix)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be a great success in a wide range of applications ranging from regular NLP-based use cases to AI agents. LLMs have been trained on a vast corpus of texts from various sources; despite the best efforts during the data pre-processing stage while training the LLMs, they may pick some undesirable information such as personally identifiable information (PII). Consequently, in recent times research in the area of Machine Unlearning (MUL) has become active, the main idea is to force LLMs to forget (unlearn) certain information (e.g., PII) without suffering from performance loss on regular tasks. In this work, we examine the robustness of the existing MUL techniques for their ability to enable leakage-proof forgetting in LLMs. In particular, we examine the effect of data transformation on forgetting, i.e., is an unlearned LLM able to recall forgotten information if there is a change in the format of the input? Our findings on the TOFU dataset highlight the necessity of using diverse data formats to quantify unlearning in LLMs more reliably.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04616v3">Beyond Answers: Transferring Reasoning Capabilities to Smaller LLMs Using Multi-Teacher Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 Accepted by WSDM 2025
    </div>
    <details class="paper-abstract">
      Transferring the reasoning capability from stronger large language models (LLMs) to smaller ones has been quite appealing, as smaller LLMs are more flexible to deploy with less expense. Among the existing solutions, knowledge distillation stands out due to its outstanding efficiency and generalization. However, existing methods suffer from several drawbacks, including limited knowledge diversity and the lack of rich contextual information. To solve the problems and facilitate the learning of compact language models, we propose TinyLLM, a new knowledge distillation paradigm to learn a small student LLM from multiple large teacher LLMs. In particular, we encourage the student LLM to not only generate the correct answers but also understand the rationales behind these answers. Given that different LLMs possess diverse reasoning skills, we guide the student model to assimilate knowledge from various teacher LLMs. We further introduce an in-context example generator and a teacher-forcing Chain-of-Thought strategy to ensure that the rationales are accurate and grounded in contextually appropriate scenarios. Extensive experiments on six datasets across two reasoning tasks demonstrate the superiority of our method. Results show that TinyLLM can outperform large teacher LLMs significantly, despite a considerably smaller model size. The source code is available at: https://github.com/YikunHan42/TinyLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15442v1">Automatic High-quality Verilog Assertion Generation through Subtask-Focused Fine-Tuned LLMs and Iterative Prompting</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Formal Property Verification (FPV), using SystemVerilog Assertions (SVA), is crucial for ensuring the completeness of design with respect to the specification. However, writing SVA is a laborious task and has a steep learning curve. In this work, we present a large language model (LLM) -based flow to automatically generate high-quality SVA from the design specification documents, named \ToolName. We introduce a novel sub-task-focused fine-tuning approach that effectively addresses functionally incorrect assertions produced by baseline LLMs, leading to a remarkable 7.3-fold increase in the number of functionally correct assertions. Recognizing the prevalence of syntax and semantic errors, we also developed an iterative refinement method that enhances the LLM's initial outputs by systematically re-prompting it to correct identified issues. This process is further strengthened by a custom compiler that generates meaningful error messages, guiding the LLM towards improved accuracy. The experiments demonstrate a 26\% increase in the number of assertions free from syntax errors using this approach, showcasing its potential to streamline the FPV process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07130v4">Harnessing LLM to Attack LLM-Guarded Text-to-Image Models</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 10 pages, 7 figures, under review
    </div>
    <details class="paper-abstract">
      To prevent Text-to-Image (T2I) models from generating unethical images, people deploy safety filters to block inappropriate drawing prompts. Previous works have employed token replacement to search adversarial prompts that attempt to bypass these filters, but they have become ineffective as nonsensical tokens fail semantic logic checks. In this paper, we approach adversarial prompts from a different perspective. We demonstrate that rephrasing a drawing intent into multiple benign descriptions of individual visual components can obtain an effective adversarial prompt. We propose a LLM-piloted multi-agent method named DACA to automatically complete intended rephrasing. Our method successfully bypasses the safety filters of DALL-E 3 and Midjourney to generate the intended images, achieving success rates of up to 76.7% and 64% in the one-time attack, and 98% and 84% in the re-use attack, respectively. We open-source our code and dataset on [this link](https://github.com/researchcode003/DACA).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16723v1">Two Heads Are Better Than One: Collaborative LLM Embodied Agents for Human-Robot Interaction</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 9 pages, 10 figures
    </div>
    <details class="paper-abstract">
      With the recent development of natural language generation models - termed as large language models (LLMs) - a potential use case has opened up to improve the way that humans interact with robot assistants. These LLMs should be able to leverage their large breadth of understanding to interpret natural language commands into effective, task appropriate and safe robot task executions. However, in reality, these models suffer from hallucinations, which may cause safety issues or deviations from the task. In other domains, these issues have been improved through the use of collaborative AI systems where multiple LLM agents can work together to collectively plan, code and self-check outputs. In this research, multiple collaborative AI systems were tested against a single independent AI agent to determine whether the success in other domains would translate into improved human-robot interaction performance. The results show that there is no defined trend between the number of agents and the success of the model. However, it is clear that some collaborative AI agent architectures can exhibit a greatly improved capacity to produce error-free code and to solve abstract problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15399v1">Less is More: Optimizing Function Calling for LLM Execution on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 Accepted at DATE 2025
    </div>
    <details class="paper-abstract">
      The advanced function-calling capabilities of foundation models open up new possibilities for deploying agents to perform complex API tasks. However, managing large amounts of data and interacting with numerous APIs makes function calling hardware-intensive and costly, especially on edge devices. Current Large Language Models (LLMs) struggle with function calling at the edge because they cannot handle complex inputs or manage multiple tools effectively. This results in low task-completion accuracy, increased delays, and higher power consumption. In this work, we introduce Less-is-More, a novel fine-tuning-free function-calling scheme for dynamic tool selection. Our approach is based on the key insight that selectively reducing the number of tools available to LLMs significantly improves their function-calling performance, execution time, and power efficiency on edge devices. Experimental results with state-of-the-art LLMs on edge hardware show agentic success rate improvements, with execution time reduced by up to 70% and power consumption by up to 40%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15396v1">The Decoy Dilemma in Online Medical Information Evaluation: A Comparative Study of Credibility Assessments by LLM and Human Judges</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Can AI be cognitively biased in automated information judgment tasks? Despite recent progresses in measuring and mitigating social and algorithmic biases in AI and large language models (LLMs), it is not clear to what extent LLMs behave "rationally", or if they are also vulnerable to human cognitive bias triggers. To address this open problem, our study, consisting of a crowdsourcing user experiment and a LLM-enabled simulation experiment, compared the credibility assessments by LLM and human judges under potential decoy effects in an information retrieval (IR) setting, and empirically examined the extent to which LLMs are cognitively biased in COVID-19 medical (mis)information assessment tasks compared to traditional human assessors as a baseline. The results, collected from a between-subject user experiment and a LLM-enabled replicate experiment, demonstrate that 1) Larger and more recent LLMs tend to show a higher level of consistency and accuracy in distinguishing credible information from misinformation. However, they are more likely to give higher ratings for misinformation due to the presence of a more salient, decoy misinformation result; 2) While decoy effect occurred in both human and LLM assessments, the effect is more prevalent across different conditions and topics in LLM judgments compared to human credibility ratings. In contrast to the generally assumed "rationality" of AI tools, our study empirically confirms the cognitive bias risks embedded in LLM agents, evaluates the decoy impact on LLMs against human credibility assessments, and thereby highlights the complexity and importance of debiasing AI agents and developing psychology-informed AI audit techniques and policies for automated judgment tasks and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06537v5">Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Recent research in neural machine translation (NMT) has shown that training on high-quality machine-generated data can outperform training on human-generated data. This work accompanies the first-ever release of a LLM-generated, MBR-decoded and QE-reranked dataset with both sentence-level and multi-sentence examples. We perform extensive experiments to demonstrate the quality of our dataset in terms of its downstream impact on NMT model performance. We find that training from scratch on our (machine-generated) dataset outperforms training on the (web-crawled) WMT'23 training dataset (which is 300 times larger), and also outperforms training on the top-quality subset of the WMT'23 training dataset. We also find that performing self-distillation by finetuning the LLM which generated this dataset outperforms the LLM's strong few-shot baseline. These findings corroborate the quality of our dataset, and demonstrate the value of high-quality machine-generated data in improving performance of NMT models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15490v3">Dynamic Intelligence Assessment: Benchmarking LLMs on the Road to AGI with a Focus on Model Confidence</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      As machine intelligence evolves, the need to test and compare the problem-solving abilities of different AI models grows. However, current benchmarks are often simplistic, allowing models to perform uniformly well and making it difficult to distinguish their capabilities. Additionally, benchmarks typically rely on static question-answer pairs that the models might memorize or guess. To address these limitations, we introduce Dynamic Intelligence Assessment (DIA), a novel methodology for testing AI models using dynamic question templates and improved metrics across multiple disciplines such as mathematics, cryptography, cybersecurity, and computer science. The accompanying dataset, DIA-Bench, contains a diverse collection of challenge templates with mutable parameters presented in various formats, including text, PDFs, compiled binaries, visual puzzles, and CTF-style cybersecurity challenges. Our framework introduces four new metrics to assess a model's reliability and confidence across multiple attempts. These metrics revealed that even simple questions are frequently answered incorrectly when posed in varying forms, highlighting significant gaps in models' reliability. Notably, API models like GPT-4o often overestimated their mathematical capabilities, while ChatGPT-4o demonstrated better performance due to effective tool usage. In self-assessment, OpenAI's o1-mini proved to have the best judgement on what tasks it should attempt to solve. We evaluated 25 state-of-the-art LLMs using DIA-Bench, showing that current models struggle with complex tasks and often display unexpectedly low confidence, even with simpler questions. The DIA framework sets a new standard for assessing not only problem-solving but also a model's adaptive intelligence and ability to assess its limitations. The dataset is publicly available on the project's page: https://github.com/DIA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03720v2">Negotiating with LLMS: Prompt Hacks, Skill Gaps, and Reasoning Deficits</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Large language models LLMs like ChatGPT have reached the 100 Mio user barrier in record time and might increasingly enter all areas of our life leading to a diverse set of interactions between those Artificial Intelligence models and humans. While many studies have discussed governance and regulations deductively from first-order principles, few studies provide an inductive, data-driven lens based on observing dialogues between humans and LLMs especially when it comes to non-collaborative, competitive situations that have the potential to pose a serious threat to people. In this work, we conduct a user study engaging over 40 individuals across all age groups in price negotiations with an LLM. We explore how people interact with an LLM, investigating differences in negotiation outcomes and strategies. Furthermore, we highlight shortcomings of LLMs with respect to their reasoning capabilities and, in turn, susceptiveness to prompt hacking, which intends to manipulate the LLM to make agreements that are against its instructions or beyond any rationality. We also show that the negotiated prices humans manage to achieve span a broad range, which points to a literacy gap in effectively interacting with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08977v2">Robustness and Confounders in the Demographic Alignment of LLMs with Human Perceptions of Offensiveness</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to exhibit demographic biases, yet few studies systematically evaluate these biases across multiple datasets or account for confounding factors. In this work, we examine LLM alignment with human annotations in five offensive language datasets, comprising approximately 220K annotations. Our findings reveal that while demographic traits, particularly race, influence alignment, these effects are inconsistent across datasets and often entangled with other factors. Confounders -- such as document difficulty, annotator sensitivity, and within-group agreement -- account for more variation in alignment patterns than demographic traits alone. Specifically, alignment increases with higher annotator sensitivity and group agreement, while greater document difficulty corresponds to reduced alignment. Our results underscore the importance of multi-dataset analyses and confounder-aware methodologies in developing robust measures of demographic bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15027v1">Time is on my sight: scene graph filtering for dynamic environment perception in an LLM-driven robot</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Robots are increasingly being used in dynamic environments like workplaces, hospitals, and homes. As a result, interactions with robots must be simple and intuitive, with robots perception adapting efficiently to human-induced changes. This paper presents a robot control architecture that addresses key challenges in human-robot interaction, with a particular focus on the dynamic creation and continuous update of the robot state representation. The architecture uses Large Language Models to integrate diverse information sources, including natural language commands, robotic skills representation, real-time dynamic semantic mapping of the perceived scene. This enables flexible and adaptive robotic behavior in complex, dynamic environments. Traditional robotic systems often rely on static, pre-programmed instructions and settings, limiting their adaptability to dynamic environments and real-time collaboration. In contrast, this architecture uses LLMs to interpret complex, high-level instructions and generate actionable plans that enhance human-robot collaboration. At its core, the system Perception Module generates and continuously updates a semantic scene graph using RGB-D sensor data, providing a detailed and structured representation of the environment. A particle filter is employed to ensure accurate object localization in dynamic, real-world settings. The Planner Module leverages this up-to-date semantic map to break down high-level tasks into sub-tasks and link them to robotic skills such as navigation, object manipulation (e.g., PICK and PLACE), and movement (e.g., GOTO). By combining real-time perception, state tracking, and LLM-driven communication and task planning, the architecture enhances adaptability, task efficiency, and human-robot collaboration in dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15279v1">Don't Mesh with Me: Generating Constructive Solid Geometry Instead of Meshes by Fine-Tuning a Code-Generation LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      While recent advancements in machine learning, such as LLMs, are revolutionizing software development and creative industries, they have had minimal impact on engineers designing mechanical parts, which remains largely a manual process. Existing approaches to generate 3D geometry most commonly use meshes as a 3D representation. While meshes are suitable for assets in video games or animations, they lack sufficient precision and adaptability for mechanical engineering purposes. This paper introduces a novel approach for the generation of 3D geometry that generates surface-based Constructive Solid Geometry (CSG) by leveraging a code-generation LLM. First, we create a dataset of 3D mechanical parts represented as code scripts by converting Boundary Representation geometry (BREP) into CSG-based Python scripts. Second, we create annotations in natural language using GPT-4. The resulting dataset is used to fine-tune a code-generation LLM. The fine-tuned LLM can complete geometries based on positional input and natural language in a plausible way, demonstrating geometric understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08989v2">Zeroth-Order Fine-Tuning of LLMs in Random Subspaces</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) has proven effective for a variety of downstream tasks. However, as LLMs grow in size, the memory demands for backpropagation become increasingly prohibitive. Zeroth-order (ZO) optimization methods offer a memory-efficient alternative by using forward passes to estimate gradients, but the variance of gradient estimates typically scales linearly with the model's parameter dimension$\unicode{x2013}$a significant issue for LLMs. In this paper, we propose the random Subspace Zeroth-order (SubZero) optimization to address the challenges posed by LLMs' high dimensionality. We introduce a low-rank perturbation tailored for LLMs that significantly reduces memory consumption while improving training performance. Additionally, we prove that our gradient estimation closely approximates the backpropagation gradient, exhibits lower variance than traditional ZO methods, and ensures convergence when combined with SGD. Experimental results show that SubZero enhances fine-tuning performance and achieves faster convergence compared to standard ZO approaches like MeZO across various language modeling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14971v1">Leveraging LLMs for Legacy Code Modernization: Challenges and Opportunities for LLM-Generated Documentation</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Abbreviated version submitted to LLM4Code 2025 (a workshop co-located with ICSE 2025), 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Legacy software systems, written in outdated languages like MUMPS and mainframe assembly, pose challenges in efficiency, maintenance, staffing, and security. While LLMs offer promise for modernizing these systems, their ability to understand legacy languages is largely unknown. This paper investigates the utilization of LLMs to generate documentation for legacy code using two datasets: an electronic health records (EHR) system in MUMPS and open-source applications in IBM mainframe Assembly Language Code (ALC). We propose a prompting strategy for generating line-wise code comments and a rubric to evaluate their completeness, readability, usefulness, and hallucination. Our study assesses the correlation between human evaluations and automated metrics, such as code complexity and reference-based metrics. We find that LLM-generated comments for MUMPS and ALC are generally hallucination-free, complete, readable, and useful compared to ground-truth comments, though ALC poses challenges. However, no automated metrics strongly correlate with comment quality to predict or measure LLM performance. Our findings highlight the limitations of current automated measures and the need for better evaluation metrics for LLM-generated documentation in legacy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14905v1">Feasibility Study for Supporting Static Malware Analysis Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming more advanced and widespread and have shown their applicability to various domains, including cybersecurity. Static malware analysis is one of the most important tasks in cybersecurity; however, it is time-consuming and requires a high level of expertise. Therefore, we conducted a demonstration experiment focusing on whether an LLM can be used to support static analysis. First, we evaluated the ability of the LLM to explain malware functionality. The results showed that the LLM can generate descriptions that cover functions with an accuracy of up to 90.9\%. In addition, we asked six static analysts to perform a pseudo static analysis task using LLM explanations to verify that the LLM can be used in practice. Through subsequent questionnaires and interviews with the participants, we also demonstrated the practical applicability of LLMs. Lastly, we summarized the problems and required functions when using an LLM as static analysis support, as well as recommendations for future research opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14896v1">Evaluating LLM Prompts for Data Augmentation in Multi-label Classification of Ecological Texts</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Ivannikov ISPRAS Open Conference (ISPRAS) 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) play a crucial role in natural language processing (NLP) tasks, improving the understanding, generation, and manipulation of human language across domains such as translating, summarizing, and classifying text. Previous studies have demonstrated that instruction-based LLMs can be effectively utilized for data augmentation to generate diverse and realistic text samples. This study applied prompt-based data augmentation to detect mentions of green practices in Russian social media. Detecting green practices in social media aids in understanding their prevalence and helps formulate recommendations for scaling eco-friendly actions to mitigate environmental issues. We evaluated several prompts for augmenting texts in a multi-label classification task, either by rewriting existing datasets using LLMs, generating new data, or combining both approaches. Our results revealed that all strategies improved classification performance compared to the models fine-tuned only on the original dataset, outperforming baselines in most cases. The best results were obtained with the prompt that paraphrased the original text while clearly indicating the relevant categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00434v2">Rapid Integration of LLMs in Healthcare Raises Ethical Concerns: An Investigation into Deceptive Patterns in Social Robots</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 7 pages, 1table, 1 figure
    </div>
    <details class="paper-abstract">
      Conversational agents are increasingly used in healthcare, and the integration of Large Language Models (LLMs) has significantly enhanced their capabilities. When integrated into social robots, LLMs offer the potential for more natural interactions. However, while LLMs promise numerous benefits, they also raise critical ethical concerns, particularly around the issue of hallucinations and deceptive patterns. In this case study, we observed a critical pattern of deceptive behavior in commercially available LLM-based care software integrated into robots. The LLM-equipped robot falsely claimed to have medication reminder functionalities. Not only did these systems assure users of their ability to manage medication schedules, but they also proactively suggested this capability, despite lacking it. This deceptive behavior poses significant risks in healthcare environments, where reliability is paramount. Our findings highlights the ethical and safety concerns surrounding the deployment of LLM-integrated robots in healthcare, emphasizing the need for oversight to prevent potentially harmful consequences for vulnerable populations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17732v1">LLM-Powered Approximate Intermittent Computing</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Batteryless IoT systems face energy constraints exacerbated by checkpointing overhead. Approximate computing offers solutions but demands manual expertise, limiting scalability. This paper presents CheckMate, an automated framework leveraging LLMs for context-aware code approximations. CheckMate integrates validation of LLM-generated approximations to ensure correct execution and employs Bayesian optimization to fine-tune approximation parameters autonomously, eliminating the need for developer input. Tested across six IoT applications, it reduces power cycles by up to 60% with an accuracy loss of just 8%, outperforming semi-automated tools like ACCEPT in speedup and accuracy. CheckMate's results establish it as a robust, user-friendly tool and a foundational step toward automated approximation frameworks for intermittent computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15735v3">Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Under Review of IEEE SaTML 2024
    </div>
    <details class="paper-abstract">
      The current cybersecurity landscape is increasingly complex, with traditional Static Application Security Testing (SAST) tools struggling to capture complex and emerging vulnerabilities due to their reliance on rule-based matching. Meanwhile, Large Language Models (LLMs) have demonstrated powerful code analysis capabilities, but their static training data and privacy risks limit their effectiveness. To overcome the limitations of both approaches, we propose LSAST, a novel approach that integrates LLMs with SAST scanners to enhance vulnerability detection. LSAST leverages a locally hostable LLM, combined with a state-of-the-art knowledge retrieval system, to provide up-to-date vulnerability insights without compromising data privacy. We set a new benchmark for static vulnerability analysis, offering a robust, privacy-conscious solution that bridges the gap between traditional scanners and advanced AI-driven analysis. Our evaluation demonstrates that incorporating SAST results into LLM analysis significantly improves detection accuracy, identifying vulnerabilities missed by conventional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13909v2">Panther: Illuminate the Sight of Multimodal LLMs with Instruction-Guided Visual Prompts</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) are closing the gap to human visual perception capability rapidly, while, still lag behind on attending to subtle images details or locating small objects precisely, etc. Common schemes to tackle these issues include deploying multiple vision encoders or operating on original high-resolution images. Few studies have concentrated on taking the textual instruction into improving visual representation, resulting in losing focus in some vision-centric tasks, a phenomenon we herein termed as Amblyopia. In this work, we introduce Panther, a MLLM that closely adheres to user instruction and locates targets of interests precisely, with the finesse of a black panther. Specifically, Panther comprises three integral components: Panther-VE, Panther-Bridge, and Panther-Decoder. Panther-VE integrates user instruction information at the early stages of the vision encoder, thereby extracting the most relevant and useful visual representations. The Panther-Bridge module, equipped with powerful filtering capabilities, significantly reduces redundant visual information, leading to a substantial savings in training costs. The Panther-Decoder is versatile and can be employed with any decoder-only architecture of LLMs without discrimination. Experimental results, particularly on vision-centric benchmarks, have demonstrated the effectiveness of Panther.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14739v1">IRLab@iKAT24: Learned Sparse Retrieval with Multi-aspect LLM Query Generation for Conversational Search</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      The Interactive Knowledge Assistant Track (iKAT) 2024 focuses on advancing conversational assistants, able to adapt their interaction and responses from personalized user knowledge. The track incorporates a Personal Textual Knowledge Base (PTKB) alongside Conversational AI tasks, such as passage ranking and response generation. Query Rewrite being an effective approach for resolving conversational context, we explore Large Language Models (LLMs), as query rewriters. Specifically, our submitted runs explore multi-aspect query generation using the MQ4CS framework, which we further enhance with Learned Sparse Retrieval via the SPLADE architecture, coupled with robust cross-encoder models. We also propose an alternative to the previous interleaving strategy, aggregating multiple aspects during the reranking phase. Our findings indicate that multi-aspect query generation is effective in enhancing performance when integrated with advanced retrieval and reranking models. Our results also lead the way for better personalization in Conversational Search, relying on LLMs to integrate personalization within query rewrite, and outperforming human rewrite performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14738v1">Universal and Context-Independent Triggers for Precise Control of LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted in applications such as automated content generation and even critical decision-making systems. However, the risk of prompt injection allows for potential manipulation of LLM outputs. While numerous attack methods have been documented, achieving full control over these outputs remains challenging, often requiring experienced attackers to make multiple attempts and depending heavily on the prompt context. Recent advancements in gradient-based white-box attack techniques have shown promise in tasks like jailbreaks and system prompt leaks. Our research generalizes gradient-based attacks to find a trigger that is (1) Universal: effective irrespective of the target output; (2) Context-Independent: robust across diverse prompt contexts; and (3) Precise Output: capable of manipulating LLM inputs to yield any specified output with high accuracy. We propose a novel method to efficiently discover such triggers and assess the effectiveness of the proposed attack. Furthermore, we discuss the substantial threats posed by such attacks to LLM-based applications, highlighting the potential for adversaries to taking over the decisions and actions made by AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14672v1">Multiverse of Greatness: Generating Story Branches with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 12 pages, 14 figures
    </div>
    <details class="paper-abstract">
      This paper presents Dynamic Context Prompting/Programming (DCP/P), a novel framework for interacting with LLMs to generate graph-based content with a dynamic context window history. While there is an existing study utilizing LLMs to generate a visual novel game, the previous study involved a manual process of output extraction and did not provide flexibility in generating a longer, coherent story. We evaluate DCP/P against our baseline, which does not provide context history to an LLM and only relies on the initial story data. Through objective evaluation, we show that simply providing the LLM with a summary leads to a subpar story compared to additionally providing the LLM with the proper context of the story. We also provide an extensive qualitative analysis and discussion. We qualitatively examine the quality of the objectively best-performing generated game from each approach. In addition, we examine biases in word choices and word sentiment of the generated content. We find a consistent observation with previous studies that LLMs are biased towards certain words, even with a different LLM family. Finally, we provide a comprehensive discussion on opportunities for future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.14971v1">Leveraging LLMs for Legacy Code Modernization: Challenges and Opportunities for LLM-Generated Documentation</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Abbreviated version submitted to LLM4Code 2025 (a workshop co-located with ICSE 2025), 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Legacy software systems, written in outdated languages like MUMPS and mainframe assembly, pose challenges in efficiency, maintenance, staffing, and security. While LLMs offer promise for modernizing these systems, their ability to understand legacy languages is largely unknown. This paper investigates the utilization of LLMs to generate documentation for legacy code using two datasets: an electronic health records (EHR) system in MUMPS and open-source applications in IBM mainframe Assembly Language Code (ALC). We propose a prompting strategy for generating line-wise code comments and a rubric to evaluate their completeness, readability, usefulness, and hallucination. Our study assesses the correlation between human evaluations and automated metrics, such as code complexity and reference-based metrics. We find that LLM-generated comments for MUMPS and ALC are generally hallucination-free, complete, readable, and useful compared to ground-truth comments, though ALC poses challenges. However, no automated metrics strongly correlate with comment quality to predict or measure LLM performance. Our findings highlight the limitations of current automated measures and the need for better evaluation metrics for LLM-generated documentation in legacy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14571v1">Assessment of LLM Responses to End-user Security Questions</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 18 pages, 1 figure, 8 tables
    </div>
    <details class="paper-abstract">
      Answering end user security questions is challenging. While large language models (LLMs) like GPT, LLAMA, and Gemini are far from error-free, they have shown promise in answering a variety of questions outside of security. We studied LLM performance in the area of end user security by qualitatively evaluating 3 popular LLMs on 900 systematically collected end user security questions. While LLMs demonstrate broad generalist ``knowledge'' of end user security information, there are patterns of errors and limitations across LLMs consisting of stale and inaccurate answers, and indirect or unresponsive communication styles, all of which impacts the quality of information received. Based on these patterns, we suggest directions for model improvement and recommend user strategies for interacting with LLMs when seeking assistance with security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16707v1">Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      The integration of experimental technologies with large language models (LLMs) is transforming scientific research, positioning AI as a versatile research assistant rather than a mere problem-solving tool. In the field of power systems, however, managing simulations -- one of the essential experimental technologies -- remains a challenge for LLMs due to their limited domain-specific knowledge, restricted reasoning capabilities, and imprecise handling of simulation parameters. To address these limitations, we propose a feedback-driven, multi-agent framework that incorporates three proposed modules: an enhanced retrieval-augmented generation (RAG) module, an improved reasoning module, and a dynamic environmental acting module with an error-feedback mechanism. Validated on 69 diverse tasks from Daline and MATPOWER, this framework achieves success rates of 93.13% and 96.85%, respectively, significantly outperforming the latest LLMs (ChatGPT 4o and o1-preview), which achieved a 27.77% success rate on standard simulation tasks and 0% on complex tasks. Additionally, our framework also supports rapid, cost-effective task execution, completing each simulation in approximately 30 seconds at an average cost of 0.014 USD for tokens. Overall, this adaptable framework lays a foundation for developing intelligent LLM-based assistants for human researchers, facilitating power system research and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14343v1">UnifiedCrawl: Aggregated Common Crawl for Affordable Adaptation of LLMs on Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) under-perform on low-resource languages due to limited training data. We present a method to efficiently collect text data for low-resource languages from the entire Common Crawl corpus. Our approach, UnifiedCrawl, filters and extracts common crawl using minimal compute resources, yielding mono-lingual datasets much larger than previously available sources. We demonstrate that leveraging this data to fine-tuning multilingual LLMs via efficient adapter methods (QLoRA) significantly boosts performance on the low-resource language, while minimizing VRAM usage. Our experiments show large improvements in language modeling perplexity and an increase in few-shot prompting scores. Our work and released source code provide an affordable approach to improve LLMs for low-resource languages using consumer hardware. Our source code is available here at https://github.com/bethelmelesse/unifiedcrawl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13009v2">LLMSteer: Improving Long-Context LLM Inference by Steering Attention on Reused Contexts</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) show impressive performance on complex tasks, they still struggle with longer contextual understanding and high computational costs. To balance efficiency and quality, we introduce LLMSteer, a fine-tuning-free framework that enhances LLMs through query-independent attention steering. Tested on popular LLMs and datasets, LLMSteer narrows the performance gap with baselines by 65.9% and reduces the runtime delay by up to 4.8x compared to recent attention steering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14283v1">CAIP: Detecting Router Misconfigurations with Context-Aware Iterative Prompting of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 12 pages, 4 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Model checkers and consistency checkers detect critical errors in router configurations, but these tools require significant manual effort to develop and maintain. LLM-based Q&A models have emerged as a promising alternative, allowing users to query partitions of configurations through prompts and receive answers based on learned patterns, thanks to transformer models pre-trained on vast datasets that provide generic configuration context for interpreting router configurations. Yet, current methods of partition-based prompting often do not provide enough network-specific context from the actual configurations to enable accurate inference. We introduce a Context-Aware Iterative Prompting (CAIP) framework that automates network-specific context extraction and optimizes LLM prompts for more precise router misconfiguration detection. CAIP addresses three challenges: (1) efficiently mining relevant context from complex configuration files, (2) accurately distinguishing between pre-defined and user-defined parameter values to prevent irrelevant context from being introduced, and (3) managing prompt context overload with iterative, guided interactions with the model. Our evaluations on synthetic and real-world configurations show that CAIP improves misconfiguration detection accuracy by more than 30% compared to partition-based LLM approaches, model checkers, and consistency checkers, uncovering over 20 previously undetected misconfigurations in real-world configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21271v2">EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      In this work, we re-formulate the model compression problem into the customized compensation problem: Given a compressed model, we aim to introduce residual low-rank paths to compensate for compression errors under customized requirements from users (e.g., tasks, compression ratios), resulting in greater flexibility in adjusting overall capacity without being constrained by specific compression formats. However, naively applying SVD to derive residual paths causes suboptimal utilization of the low-rank representation capacity. Instead, we propose Training-free Eigenspace Low-Rank Approximation (EoRA), a method that directly minimizes compression-induced errors without requiring gradient-based training, achieving fast optimization in minutes using a small amount of calibration data. EoRA projects compression errors into the eigenspace of input activations, leveraging eigenvalues to effectively prioritize the reconstruction of high-importance error components. Moreover, EoRA can be seamlessly integrated with fine-tuning and quantization to further improve effectiveness and efficiency. EoRA consistently outperforms previous methods in compensating errors for compressed LLaMA2/3 models on various tasks, such as language generation, commonsense reasoning, and math reasoning tasks (e.g., 31.31%/12.88% and 9.69% improvements on ARC-Easy/ARC-Challenge and MathQA when compensating LLaMA3-8B that is quantized to 4-bit and pruned to 2:4 sparsity). EoRA offers a scalable, training-free solution to compensate for compression errors, making it a powerful tool to deploy LLMs in various capacity and efficiency requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14256v1">Generalizing End-To-End Autonomous Driving In Real-World Environments Using Zero-Shot LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Traditional autonomous driving methods adopt a modular design, decomposing tasks into sub-tasks. In contrast, end-to-end autonomous driving directly outputs actions from raw sensor data, avoiding error accumulation. However, training an end-to-end model requires a comprehensive dataset; otherwise, the model exhibits poor generalization capabilities. Recently, large language models (LLMs) have been applied to enhance the generalization capabilities of end-to-end driving models. Most studies explore LLMs in an open-loop manner, where the output actions are compared to those of experts without direct feedback from the real world, while others examine closed-loop results only in simulations. This paper proposes an efficient architecture that integrates multimodal LLMs into end-to-end driving models operating in closed-loop settings in real-world environments. In our architecture, the LLM periodically processes raw sensor data to generate high-level driving instructions, effectively guiding the end-to-end model, even at a slower rate than the raw sensor data. This architecture relaxes the trade-off between the latency and inference quality of the LLM. It also allows us to choose from a wide variety of LLMs to improve high-level driving instructions and minimize fine-tuning costs. Consequently, our architecture reduces data collection requirements because the LLMs do not directly output actions; we only need to train a simple imitation learning model to output actions. In our experiments, the training data for the end-to-end model in a real-world environment consists of only simple obstacle configurations with one traffic cone, while the test environment is more complex and contains multiple obstacles placed in various positions. Experiments show that the proposed architecture enhances the generalization capabilities of the end-to-end model even without fine-tuning the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14214v1">Physics-Informed LLM-Agent for Automated Modulation Design in Power Electronics Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      LLM-based autonomous agents have demonstrated outstanding performance in solving complex industrial tasks. However, in the pursuit of carbon neutrality and high-performance renewable energy systems, existing AI-assisted design automation faces significant limitations in explainability, scalability, and usability. To address these challenges, we propose LP-COMDA, an LLM-based, physics-informed autonomous agent that automates the modulation design of power converters in Power Electronics Systems with minimal human supervision. Unlike traditional AI-assisted approaches, LP-COMDA contains an LLM-based planner that gathers and validates design specifications through a user-friendly chat interface. The planner then coordinates with physics-informed design and optimization tools to iteratively generate and refine modulation designs autonomously. Through the chat interface, LP-COMDA provides an explainable design process, presenting explanations and charts. Experiments show that LP-COMDA outperforms all baseline methods, achieving a 63.2% reduction in error compared to the second-best benchmark method in terms of standard mean absolute error. Furthermore, empirical studies with 20 experts conclude that design time with LP-COMDA is over 33 times faster than conventional methods, showing its significant improvement on design efficiency over the current processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18097v3">RRADistill: Distilling LLMs' Passage Ranking Ability for Long-Tail Queries Document Re-Ranking on a Search Engine</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 Accepted to EMNLP 2024 Industry Track. First two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at understanding the semantic relationships between queries and documents, even with lengthy and complex long-tail queries. These queries are challenging for feedback-based rankings due to sparse user engagement and limited feedback, making LLMs' ranking ability highly valuable. However, the large size and slow inference of LLMs necessitate the development of smaller, more efficient models (sLLMs). Recently, integrating ranking label generation into distillation techniques has become crucial, but existing methods underutilize LLMs' capabilities and are cumbersome. Our research, RRADistill: Re-Ranking Ability Distillation, propose an efficient label generation pipeline and novel sLLM training methods for both encoder and decoder models. We introduce an encoder-based method using a Term Control Layer to capture term matching signals and a decoder-based model with a ranking layer for enhanced understanding. A/B testing on a Korean-based search platform, validates the effectiveness of our approach in improving re-ranking for long-tail queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14133v1">GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 28 pages, 9 tables, 13 figures; under review at CVPR '25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05288v1">StackEval: Benchmarking LLMs in Coding Assistance</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      We present two comprehensive benchmarks to evaluate the performance of language models in coding assistance tasks, covering code writing, debugging, code review, and conceptual understanding. Our main contribution includes two curated datasets: StackEval, a large-scale benchmark derived from Stack Overflow questions, and StackUnseen, a dynamic benchmark featuring the most recent Stack Overflow content. These benchmarks offer novel insights into the capabilities and limitations of LLMs, particularly in handling new and emerging content. Additionally, we assess LLMs' proficiency as judges for coding tasks using a curated, human-annotated dataset, exploring their evaluation capabilities and potential biases, including whether they favor their own generated solutions. Our findings underscore the potential of these benchmarks to advance LLM development and application in coding assistance. To ensure reproducibility, we publicly share our datasets and evaluation code at https://github.com/ProsusAI/stack-eval .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13941v1">LLMs as Continuous Learners: Improving the Reproduction of Defective Code in Software Issues</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Reproducing buggy code is the first and crucially important step in issue resolving, as it aids in identifying the underlying problems and validating that generated patches resolve the problem. While numerous approaches have been proposed for this task, they primarily address common, widespread errors and struggle to adapt to unique, evolving errors specific to individual code repositories. To fill this gap, we propose EvoCoder, a multi-agent continuous learning framework for issue code reproduction. EvoCoder adopts a reflection mechanism that allows the LLM to continuously learn from previously resolved problems and dynamically refine its strategies to new emerging challenges. To prevent experience bloating, EvoCoder introduces a novel hierarchical experience pool that enables the model to adaptively update common and repo-specific experiences. Our experimental results show a 20\% improvement in issue reproduction rates over existing SOTA methods. Furthermore, integrating our reproduction mechanism significantly boosts the overall accuracy of the existing issue-resolving pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14502v1">Global Challenge for Safe and Secure LLMs Track 1</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      This paper introduces the Global Challenge for Safe and Secure Large Language Models (LLMs), a pioneering initiative organized by AI Singapore (AISG) and the CyberSG R&D Programme Office (CRPO) to foster the development of advanced defense mechanisms against automated jailbreaking attacks. With the increasing integration of LLMs in critical sectors such as healthcare, finance, and public administration, ensuring these models are resilient to adversarial attacks is vital for preventing misuse and upholding ethical standards. This competition focused on two distinct tracks designed to evaluate and enhance the robustness of LLM security frameworks. Track 1 tasked participants with developing automated methods to probe LLM vulnerabilities by eliciting undesirable responses, effectively testing the limits of existing safety protocols within LLMs. Participants were challenged to devise techniques that could bypass content safeguards across a diverse array of scenarios, from offensive language to misinformation and illegal activities. Through this process, Track 1 aimed to deepen the understanding of LLM vulnerabilities and provide insights for creating more resilient models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13899v1">Schemato -- An LLM for Netlist-to-Schematic Conversion</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Machine learning models are advancing circuit design, particularly in analog circuits. They typically generate netlists that lack human interpretability. This is a problem as human designers heavily rely on the interpretability of circuit diagrams or schematics to intuitively understand, troubleshoot, and develop designs. Hence, to integrate domain knowledge effectively, it is crucial to translate ML-generated netlists into interpretable schematics quickly and accurately. We propose Schemato, a large language model (LLM) for netlist-to-schematic conversion. In particular, we consider our approach in the two settings of converting netlists to .asc files for LTSpice and LATEX files for CircuiTikz schematics. Experiments on our circuit dataset show that Schemato achieves up to 93% compilation success rate for the netlist-to-LaTeX conversion task, surpassing the 26% rate scored by the state-of-the-art LLMs. Furthermore, our experiments show that Schemato generates schematics with a mean structural similarity index measure that is 3xhigher than the best performing LLMs, therefore closer to the reference human design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13874v1">Next-Generation Phishing: How LLM Agents Empower Cyber Attackers</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      The escalating threat of phishing emails has become increasingly sophisticated with the rise of Large Language Models (LLMs). As attackers exploit LLMs to craft more convincing and evasive phishing emails, it is crucial to assess the resilience of current phishing defenses. In this study we conduct a comprehensive evaluation of traditional phishing detectors, such as Gmail Spam Filter, Apache SpamAssassin, and Proofpoint, as well as machine learning models like SVM, Logistic Regression, and Naive Bayes, in identifying both traditional and LLM-rephrased phishing emails. We also explore the emerging role of LLMs as phishing detection tools, a method already adopted by companies like NTT Security Holdings and JPMorgan Chase. Our results reveal notable declines in detection accuracy for rephrased emails across all detectors, highlighting critical weaknesses in current phishing defenses. As the threat landscape evolves, our findings underscore the need for stronger security controls and regulatory oversight on LLM-generated content to prevent its misuse in creating advanced phishing attacks. This study contributes to the development of more effective Cyber Threat Intelligence (CTI) by leveraging LLMs to generate diverse phishing variants that can be used for data augmentation, harnessing the power of LLMs to enhance phishing detection, and paving the way for more robust and adaptable threat detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13865v1">HARec: Hyperbolic Graph-LLM Alignment for Exploration and Exploitation in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Modern recommendation systems often create information cocoons, limiting users' exposure to diverse content. To enhance user experience, a crucial challenge is developing systems that can balance content exploration and exploitation, allowing users to adjust their recommendation preferences. Intuitively, this balance can be achieved through a tree-structured representation, where depth search facilitates exploitation and breadth search enables exploration. However, current works face two challenges to achieve this target: (1) Euclidean methods fail to fully capture hierarchical structures and lack flexibility in balancing exploration-exploitation, while (2) hyperbolic approaches, despite better hierarchical modeling, suffer from insufficient semantic alignment due to their reliance on Euclidean text encoders. To address these challenges, we propose HARec, a hyperbolic representation learning framework that jointly aligns user-item collaborative information with textual descriptions in hyperbolic space. Our framework introduces two key technique novelty: (1) a hierarchical-aware graph-llm alignment mechanism that enables better hierarchical representation, and (2) a hyperbolic hierarchical tree structure that facilitates user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HARec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09955v2">Instruction-Guided Editing Controls for Images and Multimedia: A Survey in LLM era</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 Fixed a serious error in author information
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) and multimodal learning has transformed digital content creation and manipulation. Traditional visual editing tools require significant expertise, limiting accessibility. Recent strides in instruction-based editing have enabled intuitive interaction with visual content, using natural language as a bridge between user intent and complex editing operations. This survey provides an overview of these techniques, focusing on how LLMs and multimodal models empower users to achieve precise visual modifications without deep technical knowledge. By synthesizing over 100 publications, we explore methods from generative adversarial networks to diffusion models, examining multimodal integration for fine-grained content control. We discuss practical applications across domains such as fashion, 3D scene manipulation, and video synthesis, highlighting increased accessibility and alignment with human intuition. Our survey compares existing literature, emphasizing LLM-empowered editing, and identifies key challenges to stimulate further research. We aim to democratize powerful visual editing across various industries, from entertainment to education. Interested readers are encouraged to access our repository at https://github.com/tamlhp/awesome-instruction-editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13820v1">InstCache: A Predictive Cache for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Large language models are revolutionizing every aspect of human life. However, the unprecedented power comes at the cost of significant computing intensity, suggesting long latency and large energy footprint. Key-Value Cache and Semantic Cache have been proposed as a solution to the above problem, but both suffer from limited scalability due to significant memory cost for each token or instruction embeddings. Motivated by the observations that most instructions are short, repetitive and predictable by LLMs, we propose to predict user-instructions by an instruction-aligned LLM and store them in a predictive cache, so-called InstCache. We introduce an instruction pre-population algorithm based on the negative log likelihood of instructions, determining the cache size with regard to the hit rate. The proposed InstCache is efficiently implemented as a hash table with minimal lookup latency for deployment. Experimental results show that InstCache can achieve up to 51.34% hit rate on LMSys dataset, which corresponds to a 2x speedup, at a memory cost of only 4.5GB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14497v1">Star-Agents: Automatic Data Optimization with LLM Agents for Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      The efficacy of large language models (LLMs) on downstream tasks usually hinges on instruction tuning, which relies critically on the quality of training data. Unfortunately, collecting high-quality and diverse data is both expensive and time-consuming. To mitigate this issue, we propose a novel Star-Agents framework, which automates the enhancement of data quality across datasets through multi-agent collaboration and assessment. The framework adopts a three-pronged strategy. It initially generates diverse instruction data with multiple LLM agents through a bespoke sampling method. Subsequently, the generated data undergo a rigorous evaluation using a dual-model method that assesses both difficulty and quality. Finaly, the above process evolves in a dynamic refinement phase, where more effective LLMs are prioritized, enhancing the overall data quality. Our empirical studies, including instruction tuning experiments with models such as Pythia and LLaMA, demonstrate the effectiveness of the proposed framework. Optimized datasets have achieved substantial improvements, with an average increase of 12% and notable gains in specific metrics, such as a 40% improvement in Fermi, as evidenced by benchmarks like MT-bench, Vicuna bench, and WizardLM testset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13779v1">NewsInterview: a Dataset and a Playground to Evaluate LLMs' Ground Gap via Informational Interviews</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in generating coherent text but often struggle with grounding language and strategic dialogue. To address this gap, we focus on journalistic interviews, a domain rich in grounding communication and abundant in data. We curate a dataset of 40,000 two-person informational interviews from NPR and CNN, and reveal that LLMs are significantly less likely than human interviewers to use acknowledgements and to pivot to higher-level questions. Realizing that a fundamental deficit exists in multi-turn planning and strategic thinking, we develop a realistic simulated environment, incorporating source personas and persuasive elements, in order to facilitate the development of agents with longer-horizon rewards. Our experiments show that while source LLMs mimic human behavior in information sharing, interviewer LLMs struggle with recognizing when questions are answered and engaging persuasively, leading to suboptimal information extraction across model size and capability. These findings underscore the need for enhancing LLMs' strategic dialogue capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11853v2">Chat Bankman-Fried: an Exploration of LLM Alignment in Finance</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) have renewed concerns about AI alignment - the consistency between human and AI goals and values. As various jurisdictions enact legislation on AI safety, the concept of alignment must be defined and measured across different domains. This paper proposes an experimental framework to assess whether LLMs adhere to ethical and legal standards in the relatively unexplored context of finance. We prompt nine LLMs to impersonate the CEO of a financial institution and test their willingness to misuse customer assets to repay outstanding corporate debt. Beginning with a baseline configuration, we adjust preferences, incentives and constraints, analyzing the impact of each adjustment with logistic regression. Our findings reveal significant heterogeneity in the baseline propensity for unethical behavior of LLMs. Factors such as risk aversion, profit expectations, and regulatory environment consistently influence misalignment in ways predicted by economic theory, although the magnitude of these effects varies across LLMs. This paper highlights both the benefits and limitations of simulation-based, ex post safety testing. While it can inform financial authorities and institutions aiming to ensure LLM safety, there is a clear trade-off between generality and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13768v1">An Evaluation-Driven Approach to Designing LLM Agents: Process and Architecture</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has enabled the development of LLM agents capable of autonomously achieving under-specified goals and continuously evolving through post-deployment improvement, sometimes without requiring code or model updates. Conventional approaches, such as pre-defined test cases and code/model redevelopment pipelines, are inadequate for addressing the unique challenges of LLM agent development, particularly in terms of quality and risk control. This paper introduces an evaluation-driven design approach, inspired by test-driven development, to address these challenges. Through a multivocal literature review (MLR), we synthesize existing LLM evaluation methods and propose a novel process model and reference architecture specifically designed for LLM agents. The proposed approach integrates online and offline evaluations to support adaptive runtime adjustments and systematic offline redevelopment, improving runtime pipelines, artifacts, system architecture, and LLMs by continuously incorporating evaluation results, including fine-grained feedback from human and AI evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13760v1">A Framework for Evaluating LLMs Under Task Indeterminacy</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 To Appear in NeurIPS 2024 Workshops on Evaluating Evaluations (EvalEval) and Statistical Foundations of LLMs and Foundation Models (SFLLM)
    </div>
    <details class="paper-abstract">
      Large language model (LLM) evaluations often assume there is a single correct response -- a gold label -- for each item in the evaluation corpus. However, some tasks can be ambiguous -- i.e., they provide insufficient information to identify a unique interpretation -- or vague -- i.e., they do not clearly indicate where to draw the line when making a determination. Both ambiguity and vagueness can cause task indeterminacy -- the condition where some items in the evaluation corpus have more than one correct response. In this paper, we develop a framework for evaluating LLMs under task indeterminacy. Our framework disentangles the relationships between task specification, human ratings, and LLM responses in the LLM evaluation pipeline. Using our framework, we conduct a synthetic experiment showing that evaluations that use the "gold label" assumption underestimate the true performance. We also provide a method for estimating an error-adjusted performance interval given partial knowledge about indeterminate items in the evaluation corpus. We conclude by outlining implications of our work for the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13757v1">AttentionBreaker: Adaptive Evolutionary Optimization for Unmasking Vulnerabilities in LLMs through Bit-Flip Attacks</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17722v1">When IoT Meet LLMs: Applications and Challenges</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData), 10 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have positively and efficiently transformed workflows in many domains. One such domain with significant potential for LLM integration is the Internet of Things (IoT), where this integration brings new opportunities for improved decision making and system interaction. In this paper, we explore the various roles of LLMs in IoT, with a focus on their reasoning capabilities. We show how LLM-IoT integration can facilitate advanced decision making and contextual understanding in a variety of IoT scenarios. Furthermore, we explore the integration of LLMs with edge, fog, and cloud computing paradigms, and show how this synergy can optimize resource utilization, enhance real-time processing, and provide scalable solutions for complex IoT applications. To the best of our knowledge, this is the first comprehensive study covering IoT-LLM integration between edge, fog, and cloud systems. Additionally, we propose a novel system model for industrial IoT applications that leverages LLM-based collective intelligence to enable predictive maintenance and condition monitoring. Finally, we highlight key challenges and open issues that provide insights for future research in the field of LLM-IoT integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13738v1">Assessing Gender Bias in LLMs: Comparing LLM Outputs with Human Perceptions and Official Statistics</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 under review for Coling conference
    </div>
    <details class="paper-abstract">
      This study investigates gender bias in large language models (LLMs) by comparing their gender perception to that of human respondents, U.S. Bureau of Labor Statistics data, and a 50% no-bias benchmark. We created a new evaluation set using occupational data and role-specific sentences. Unlike common benchmarks included in LLM training data, our set is newly developed, preventing data leakage and test set contamination. Five LLMs were tested to predict the gender for each role using single-word answers. We used Kullback-Leibler (KL) divergence to compare model outputs with human perceptions, statistical data, and the 50% neutrality benchmark. All LLMs showed significant deviation from gender neutrality and aligned more with statistical data, still reflecting inherent biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02306v2">On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      As LLMs become more widely deployed, there is increasing interest in directly optimizing for feedback from end users (e.g. thumbs up) in addition to feedback from paid annotators. However, training to maximize human feedback creates a perverse incentive structure for the AI to resort to manipulative or deceptive tactics to obtain positive feedback from users who are vulnerable to such strategies. We study this phenomenon by training LLMs with Reinforcement Learning with simulated user feedback in environments of practical LLM usage. In our settings, we find that: 1) Extreme forms of "feedback gaming" such as manipulation and deception are learned reliably; 2) Even if only 2% of users are vulnerable to manipulative strategies, LLMs learn to identify and target them while behaving appropriately with other users, making such behaviors harder to detect; 3) To mitigate this issue, it may seem promising to leverage continued safety training or LLM-as-judges during training to filter problematic outputs. Instead, we found that while such approaches help in some of our settings, they backfire in others, sometimes even leading to subtler manipulative behaviors. We hope our results can serve as a case study which highlights the risks of using gameable feedback sources -- such as user feedback -- as a target for RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13547v1">SpecTool: A Benchmark for Characterizing Errors in Tool-Use LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Evaluating the output of Large Language Models (LLMs) is one of the most critical aspects of building a performant compound AI system. Since the output from LLMs propagate to downstream steps, identifying LLM errors is crucial to system performance. A common task for LLMs in AI systems is tool use. While there are several benchmark environments for evaluating LLMs on this task, they typically only give a success rate without any explanation of the failure cases. To solve this problem, we introduce SpecTool, a new benchmark to identify error patterns in LLM output on tool-use tasks. Our benchmark data set comprises of queries from diverse environments that can be used to test for the presence of seven newly characterized error patterns. Using SPECTOOL , we show that even the most prominent LLMs exhibit these error patterns in their outputs. Researchers can use the analysis and insights from SPECTOOL to guide their error mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13543v1">BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Vision Language Models (VLMs) possess extensive knowledge and exhibit promising reasoning abilities; however, they still struggle to perform well in complex, dynamic environments. Real-world tasks require handling intricate interactions, advanced spatial reasoning, long-term planning, and continuous exploration of new strategies-areas in which we lack effective methodologies for comprehensively evaluating these capabilities. To address this gap, we introduce BALROG, a novel benchmark designed to assess the agentic capabilities of LLMs and VLMs through a diverse set of challenging games. Our benchmark incorporates a range of existing reinforcement learning environments with varying levels of difficulty, including tasks that are solvable by non-expert humans in seconds to extremely challenging ones that may take years to master (e.g., the NetHack Learning Environment). We devise fine-grained metrics to measure performance and conduct an extensive evaluation of several popular open-source and closed-source LLMs and VLMs. Our findings indicate that while current models achieve partial success in the easier games, they struggle significantly with more challenging tasks. Notably, we observe severe deficiencies in vision-based decision-making, as models perform worse when visual representations of the environments are provided. We release BALROG as an open and user-friendly benchmark to facilitate future research and development in the agentic community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13410v1">A Survey On Enhancing Reinforcement Learning in Complex Environments: Insights from Human and LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is one of the active fields in machine learning, demonstrating remarkable potential in tackling real-world challenges. Despite its promising prospects, this methodology has encountered with issues and challenges, hindering it from achieving the best performance. In particular, these approaches lack decent performance when navigating environments and solving tasks with large observation space, often resulting in sample-inefficiency and prolonged learning times. This issue, commonly referred to as the curse of dimensionality, complicates decision-making for RL agents, necessitating a careful balance between attention and decision-making. RL agents, when augmented with human or large language models' (LLMs) feedback, may exhibit resilience and adaptability, leading to enhanced performance and accelerated learning. Such feedback, conveyed through various modalities or granularities including natural language, serves as a guide for RL agents, aiding them in discerning relevant environmental cues and optimizing decision-making processes. In this survey paper, we mainly focus on problems of two-folds: firstly, we focus on humans or an LLMs assistance, investigating the ways in which these entities may collaborate with the RL agent in order to foster optimal behavior and expedite learning; secondly, we delve into the research papers dedicated to addressing the intricacies of environments characterized by large observation space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13409v1">Unification of Balti and trans-border sister dialects in the essence of LLMs and AI Technology</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 Accepted by IEEE conference ISCSLP 2024
    </div>
    <details class="paper-abstract">
      The language called Balti belongs to the Sino-Tibetan, specifically the Tibeto-Burman language family. It is understood with variations, across populations in India, China, Pakistan, Nepal, Tibet, Burma, and Bhutan, influenced by local cultures and producing various dialects. Considering the diverse cultural, socio-political, religious, and geographical impacts, it is important to step forward unifying the dialects, the basis of common root, lexica, and phonological perspectives, is vital. In the era of globalization and the increasingly frequent developments in AI technology, understanding the diversity and the efforts of dialect unification is important to understanding commonalities and shortening the gaps impacted by unavoidable circumstances. This article analyzes and examines how artificial intelligence AI in the essence of Large Language Models LLMs, can assist in analyzing, documenting, and standardizing the endangered Balti Language, based on the efforts made in different dialects so far.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13405v1">On the Way to LLM Personalization: Learning to Remember User Conversations</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 16 pages, 6 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have quickly become an invaluable assistant for a variety of tasks. However, their effectiveness is constrained by their ability to tailor responses to human preferences and behaviors via personalization. Prior work in LLM personalization has largely focused on style transfer or incorporating small factoids about the user, as knowledge injection remains an open challenge. In this paper, we explore injecting knowledge of prior conversations into LLMs to enable future work on less redundant, personalized conversations. We identify two real-world constraints: (1) conversations are sequential in time and must be treated as such during training, and (2) per-user personalization is only viable in parameter-efficient settings. To this aim, we propose PLUM, a pipeline performing data augmentation for up-sampling conversations as question-answer pairs, that are then used to finetune a low-rank adaptation adapter with a weighted cross entropy loss. Even in this first exploration of the problem, we perform competitively with baselines such as RAG, attaining an accuracy of 81.5% across 100 conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13627v1">CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Cryptographic protocols play a fundamental role in securing modern digital infrastructure, but they are often deployed without prior formal verification. This could lead to the adoption of distributed systems vulnerable to attack vectors. Formal verification methods, on the other hand, require complex and time-consuming techniques that lack automatization. In this paper, we introduce a benchmark to assess the ability of Large Language Models (LLMs) to autonomously identify vulnerabilities in new cryptographic protocols through interaction with Tamarin: a theorem prover for protocol verification. We created a manually validated dataset of novel, flawed, communication protocols and designed a method to automatically verify the vulnerabilities found by the AI agents. Our results about the performances of the current frontier models on the benchmark provides insights about the possibility of cybersecurity applications by integrating LLMs with symbolic reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13269v1">Towards Specification-Driven LLM-Based Generation of Embedded Automotive Software</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 21 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The paper studies how code generation by LLMs can be combined with formal verification to produce critical embedded software. The first contribution is a general framework, spec2code, in which LLMs are combined with different types of critics that produce feedback for iterative backprompting and fine-tuning. The second contribution presents a first feasibility study, where a minimalistic instantiation of spec2code, without iterative backprompting and fine-tuning, is empirically evaluated using three industrial case studies from the heavy vehicle manufacturer Scania. The goal is to automatically generate industrial-quality code from specifications only. Different combinations of formal ACSL specifications and natural language specifications are explored. The results indicate that formally correct code can be generated even without the application of iterative backprompting and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13159v1">Hard-Synth: Synthesizing Diverse Hard Samples for ASR using Zero-Shot TTS and LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Text-to-speech (TTS) models have been widely adopted to enhance automatic speech recognition (ASR) systems using text-only corpora, thereby reducing the cost of labeling real speech data. Existing research primarily utilizes additional text data and predefined speech styles supported by TTS models. In this paper, we propose Hard-Synth, a novel ASR data augmentation method that leverages large language models (LLMs) and advanced zero-shot TTS. Our approach employs LLMs to generate diverse in-domain text through rewriting, without relying on additional text data. Rather than using predefined speech styles, we introduce a hard prompt selection method with zero-shot TTS to clone speech styles that the ASR model finds challenging to recognize. Experiments demonstrate that Hard-Synth significantly enhances the Conformer model, achieving relative word error rate (WER) reductions of 6.5\%/4.4\% on LibriSpeech dev/test-other subsets. Additionally, we show that Hard-Synth is data-efficient and capable of reducing bias in ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02926v3">Demystifying RCE Vulnerabilities in LLM-Integrated Apps</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      LLMs show promise in transforming software development, with a growing interest in integrating them into more intelligent apps. Frameworks like LangChain aid LLM-integrated app development, offering code execution utility/APIs for custom actions. However, these capabilities theoretically introduce Remote Code Execution (RCE) vulnerabilities, enabling remote code execution through prompt injections. No prior research systematically investigates these frameworks' RCE vulnerabilities or their impact on applications and exploitation consequences. Therefore, there is a huge research gap in this field. In this study, we propose LLMSmith to detect, validate and exploit the RCE vulnerabilities in LLM-integrated frameworks and apps. To achieve this goal, we develop two novel techniques, including 1) a lightweight static analysis to examine LLM integration mechanisms, and construct call chains to identify RCE vulnerabilities in frameworks; 2) a systematical prompt-based exploitation method to verify and exploit the found vulnerabilities in LLM-integrated apps. This technique involves various strategies to control LLM outputs, trigger RCE vulnerabilities and launch subsequent attacks. Our research has uncovered a total of 20 vulnerabilities in 11 LLM-integrated frameworks, comprising 19 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. Of these, 17 have been confirmed by the framework developers, with 11 vulnerabilities being assigned CVE IDs. For the 51 apps potentially affected by RCE, we successfully executed attacks on 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. Furthermore, we conduct a comprehensive analysis of these vulnerabilities and construct practical attacks to demonstrate the hazards in reality. Last, we propose several mitigation measures for both framework and app developers to counteract such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13045v1">Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 Submitted to WWW 2025
    </div>
    <details class="paper-abstract">
      Effective query-item relevance modeling is pivotal for enhancing user experience and safeguarding user satisfaction in e-commerce search systems. Recently, benefiting from the vast inherent knowledge, Large Language Model (LLM) approach demonstrates strong performance and long-tail generalization ability compared with previous neural-based specialized relevance learning methods. Though promising, current LLM-based methods encounter the following inadequacies in practice: First, the massive parameters and computational demands make it difficult to be deployed online. Second, distilling LLM models to online models is a feasible direction, but the LLM relevance modeling is a black box, and its rich intrinsic knowledge is difficult to extract and apply online. To improve the interpretability of LLM and boost the performance of online relevance models via LLM, we propose an Explainable LLM-driven Multi-dimensional Distillation framework for e-commerce relevance learning, which comprises two core components: (1) An Explainable LLM for relevance modeling (ELLM-rele), which decomposes the relevance learning into intermediate steps and models relevance learning as a Chain-of-Thought (CoT) reasoning, thereby enhancing both interpretability and performance of LLM. (2) A Multi-dimensional Knowledge Distillation (MKD) architecture that transfers the knowledge of ELLM-rele to current deployable interaction-based and representation-based student models from both the relevance score distribution and CoT reasoning aspects. Through distilling the probabilistic and CoT reasoning knowledge, MKD improves both the semantic interaction and long-tail generalization abilities of student models. Extensive offline evaluations and online experiments on Taobao search ad scene demonstrate that our proposed framework significantly enhances e-commerce relevance learning performance and user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13008v1">Evaluating LLMs Capabilities Towards Understanding Social Dynamics</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 To appear in ASONAM 24 proceedings
    </div>
    <details class="paper-abstract">
      Social media discourse involves people from different backgrounds, beliefs, and motives. Thus, often such discourse can devolve into toxic interactions. Generative Models, such as Llama and ChatGPT, have recently exploded in popularity due to their capabilities in zero-shot question-answering. Because these models are increasingly being used to ask questions of social significance, a crucial research question is whether they can understand social media dynamics. This work provides a critical analysis regarding generative LLM's ability to understand language and dynamics in social contexts, particularly considering cyberbullying and anti-cyberbullying (posts aimed at reducing cyberbullying) interactions. Specifically, we compare and contrast the capabilities of different large language models (LLMs) to understand three key aspects of social dynamics: language, directionality, and the occurrence of bullying/anti-bullying messages. We found that while fine-tuned LLMs exhibit promising results in some social media understanding tasks (understanding directionality), they presented mixed results in others (proper paraphrasing and bullying/anti-bullying detection). We also found that fine-tuning and prompt engineering mechanisms can have positive effects in some tasks. We believe that a understanding of LLM's capabilities is crucial to design future models that can be effectively used in social applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12103v2">Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language model unlearning aims to remove harmful information that LLMs have learnt to prevent their use for malicious purposes. LLMU and RMU have been proposed as two methods for LLM unlearning, achieving impressive results on unlearning benchmarks. We study in detail the efficacy of these methods by evaluating their impact on general model capabilities on the WMDP benchmark as well as a biology benchmark we create. Our experiments show that RMU generally leads to better preservation of model capabilities, for similar or better unlearning. We further test the robustness of these methods and find that doing 5-shot prompting or rephrasing the question in simple ways can lead to an over ten-fold increase in accuracy on unlearning benchmarks. Finally, we show that training on unrelated data can almost completely recover pre-unlearning performance, demonstrating that these methods fail at truly unlearning. The code is available at: https://github.com/JaiDoshi/Knowledge-Erasure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18003v4">Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 Published on the First Conference on Language Modeling (COLM 2024)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), epitomized by ChatGPT's release in late 2022, have revolutionized various industries with their advanced language comprehension. However, their efficiency is challenged by the Transformer architecture's struggle with handling long texts. KV Cache has emerged as a pivotal solution to this issue, converting the time complexity of token generation from quadratic to linear, albeit with increased GPU memory overhead proportional to conversation length. With the development of the LLM community and academia, various KV Cache compression methods have been proposed. In this review, we dissect the various properties of KV Cache and elaborate on various methods currently used to optimize the KV Cache space usage of LLMs. These methods span the pre-training phase, deployment phase, and inference phase, and we summarize the commonalities and differences among these methods. Additionally, we list some metrics for evaluating the long-text capabilities of large language models, from both efficiency and capability perspectives. Our review thus sheds light on the evolving landscape of LLM optimization, offering insights into future advancements in this dynamic field. Links to the papers mentioned in this review can be found in our Github Repo https://github.com/zcli-charlie/Awesome-KV-Cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14484v1">Robust Planning with Compound LLM Architectures: An LLM-Modulo Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Previous work has attempted to boost Large Language Model (LLM) performance on planning and scheduling tasks through a variety of prompt engineering techniques. While these methods can work within the distributions tested, they are neither robust nor predictable. This limitation can be addressed through compound LLM architectures where LLMs work in conjunction with other components to ensure reliability. In this paper, we present a technical evaluation of a compound LLM architecture--the LLM-Modulo framework. In this framework, an LLM is paired with a complete set of sound verifiers that validate its output, re-prompting it if it fails. This approach ensures that the system can never output any fallacious output, and therefore that every output generated is guaranteed correct--something previous techniques have not been able to claim. Our results, evaluated across four scheduling domains, demonstrate significant performance gains with the LLM-Modulo framework using various models. Additionally, we explore modifications to the base configuration of the framework and assess their impact on overall system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.14484v1">Robust Planning with Compound LLM Architectures: An LLM-Modulo Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Previous work has attempted to boost Large Language Model (LLM) performance on planning and scheduling tasks through a variety of prompt engineering techniques. While these methods can work within the distributions tested, they are neither robust nor predictable. This limitation can be addressed through compound LLM architectures where LLMs work in conjunction with other components to ensure reliability. In this paper, we present a technical evaluation of a compound LLM architecture--the LLM-Modulo framework. In this framework, an LLM is paired with a complete set of sound verifiers that validate its output, re-prompting it if it fails. This approach ensures that the system can never output any fallacious output, and therefore that every output generated is guaranteed correct--something previous techniques have not been able to claim. Our results, evaluated across four scheduling domains, demonstrate significant performance gains with the LLM-Modulo framework using various models. Additionally, we explore modifications to the base configuration of the framework and assess their impact on overall system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12930v1">LEDRO: LLM-Enhanced Design Space Reduction and Optimization for Analog Circuits</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Traditional approaches for designing analog circuits are time-consuming and require significant human expertise. Existing automation efforts using methods like Bayesian Optimization (BO) and Reinforcement Learning (RL) are sub-optimal and costly to generalize across different topologies and technology nodes. In our work, we introduce a novel approach, LEDRO, utilizing Large Language Models (LLMs) in conjunction with optimization techniques to iteratively refine the design space for analog circuit sizing. LEDRO is highly generalizable compared to other RL and BO baselines, eliminating the need for design annotation or model training for different topologies or technology nodes. We conduct a comprehensive evaluation of our proposed framework and baseline on 22 different Op-Amp topologies across four FinFET technology nodes. Results demonstrate the superior performance of LEDRO as it outperforms our best baseline by an average of 13% FoM improvement with 2.15x speed-up on low complexity Op-Amps and 48% FoM improvement with 1.7x speed-up on high complexity Op-Amps. This highlights LEDRO's effective performance, efficiency, and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13082v2">Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Recent successes in natural language processing have led to the proliferation of large language models (LLMs) by multiple providers. Each LLM offering has different inference accuracy, monetary cost, and latency, and their accuracy further depends on the exact wording of the question (i.e., the specific prompt). At the same time, users often have a limit on monetary budget and latency to answer all their questions, and they do not know which LLMs to choose for each question to meet their accuracy and long term budget requirements. To navigate this rich design space, we propose TREACLE ($\underline{T}$hrifty $\underline{Rea}$soning via $\underline{C}$ontext-Aware $\underline{L}$LM and Prompt S$\underline{e}$lection), a reinforcement learning policy that jointly selects the model and prompting scheme while respecting the user's monetary cost and latency constraints. TREACLE uses the problem context, including question text embeddings (reflecting the type or difficulty of a query) and the response history (reflecting the consistency of previous responses) to make smart decisions. Our evaluations on standard reasoning datasets (GSM8K, CSQA, and LLC) with various LLMs and prompts show that TREACLE enables cost savings of up to 85% compared to baselines, while maintaining high accuracy. Importantly, it provides the user with the ability to gracefully trade off accuracy for cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12882v1">ProSec: Fortifying Code LLMs with Proactive Security Alignment</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 The first two authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Recent advances in code-specific large language models (LLMs) have greatly enhanced code generation and refinement capabilities. However, the safety of code LLMs remains under-explored, posing potential risks as insecure code generated by these models may introduce vulnerabilities into real-world systems. Previous work proposes to collect security-focused instruction-tuning dataset from real-world vulnerabilities. It is constrained by the data sparsity of vulnerable code, and has limited applicability in the iterative post-training workflows of modern LLMs. In this paper, we propose ProSec, a novel proactive security alignment approach designed to align code LLMs with secure coding practices. ProSec systematically exposes the vulnerabilities in a code LLM by synthesizing error-inducing coding scenarios from Common Weakness Enumerations (CWEs), and generates fixes to vulnerable code snippets, allowing the model to learn secure practices through advanced preference learning objectives. The scenarios synthesized by ProSec triggers 25 times more vulnerable code than a normal instruction-tuning dataset, resulting in a security-focused alignment dataset 7 times larger than the previous work. Experiments show that models trained with ProSec is 29.2% to 35.5% more secure compared to previous work, with a marginal negative effect of less than 2 percentage points on model's utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07447v2">The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      The growing usage of Large Language Models (LLMs) highlights the demands and challenges in scalable LLM inference systems, affecting deployment and development processes. On the deployment side, there is a lack of comprehensive analysis on the conditions under which a particular scheduler performs better or worse, with performance varying substantially across different schedulers, hardware, models, and workloads. Manually testing each configuration on GPUs can be prohibitively expensive. On the development side, unpredictable performance and unknown upper limits can lead to inconclusive trial-and-error processes, consuming resources on ideas that end up ineffective. To address these challenges, we introduce INFERMAX, an analytical framework that uses inference cost models to compare various schedulers, including an optimal scheduler formulated as a constraint satisfaction problem (CSP) to establish an upper bound on performance. Our framework offers in-depth analysis and raises essential questions, challenging assumptions and exploring opportunities for more efficient scheduling. Notably, our findings indicate that preempting requests can reduce GPU costs by 30% compared to avoiding preemptions at all. We believe our methods and insights will facilitate the cost-effective deployment and development of scalable, efficient inference systems and pave the way for cost-based scheduling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14483v1">Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Deciding which large language model (LLM) to use is a complex challenge. Pairwise ranking has emerged as a new method for evaluating human preferences for LLMs. This approach entails humans evaluating pairs of model outputs based on a predefined criterion. By collecting these comparisons, a ranking can be constructed using methods such as Elo. However, applying these algorithms as constructed in the context of LLM evaluation introduces several challenges. In this paper, we explore the effectiveness of ranking systems for head-to-head comparisons of LLMs. We formally define a set of fundamental principles for effective ranking and conduct a series of extensive evaluations on the robustness of several ranking algorithms in the context of LLMs. Our analysis uncovers key insights into the factors that affect ranking accuracy and efficiency, offering guidelines for selecting the most appropriate methods based on specific evaluation contexts and resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12712v1">Enhancing Multi-Class Disease Classification: Neoplasms, Cardiovascular, Nervous System, and Digestive Disorders Using Advanced LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 7 Pages, 4 tables and 11 figures. Under review in a IEEE conference
    </div>
    <details class="paper-abstract">
      In this research, we explored the improvement in terms of multi-class disease classification via pre-trained language models over Medical-Abstracts-TC-Corpus that spans five medical conditions. We excluded non-cancer conditions and examined four specific diseases. We assessed four LLMs, BioBERT, XLNet, and BERT, as well as a novel base model (Last-BERT). BioBERT, which was pre-trained on medical data, demonstrated superior performance in medical text classification (97% accuracy). Surprisingly, XLNet followed closely (96% accuracy), demonstrating its generalizability across domains even though it was not pre-trained on medical data. LastBERT, a custom model based on the lighter version of BERT, also proved competitive with 87.10% accuracy (just under BERT's 89.33%). Our findings confirm the importance of specialized models such as BioBERT and also support impressions around more general solutions like XLNet and well-tuned transformer architectures with fewer parameters (in this case, LastBERT) in medical domain tasks.
    </details>
</div>
