# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18101v3">A Cost-Benefit Analysis of On-Premise Large Language Model Deployment: Breaking Even with Commercial LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming increasingly widespread. Organizations that want to use AI for productivity now face an important decision. They can subscribe to commercial LLM services or deploy models on their own infrastructure. Cloud services from providers such as OpenAI, Anthropic, and Google are attractive because they provide easy access to state-of-the-art models and are easy to scale. However, concerns about data privacy, the difficulty of switching service providers, and long-term operating costs have driven interest in local deployment of open-source models. This paper presents a cost-benefit analysis framework to help organizations determine when on-premise LLM deployment becomes economically viable compared to commercial subscription services. We consider the hardware requirements, operational expenses, and performance benchmarks of the latest open-source models, including Qwen, Llama, Mistral, and etc. Then we compare the total cost of deploying these models locally with the major cloud providers subscription fee. Our findings provide an estimated breakeven point based on usage levels and performance needs. These results give organizations a practical framework for planning their LLM strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07800v1">From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) based agents have demonstrated remarkable potential in autonomous task-solving across complex, open-ended environments. A promising approach for improving the reasoning capabilities of LLM agents is to better utilize prior experiences in guiding current decisions. However, LLMs acquire experience either through implicit memory via training, which suffers from catastrophic forgetting and limited interpretability, or explicit memory via prompting, which lacks adaptability. In this paper, we introduce a novel agent-centric, trainable, multi-layered graph memory framework and evaluate how context memory enhances the ability of LLMs to utilize parametric information. The graph abstracts raw agent trajectories into structured decision paths in a state machine and further distills them into high-level, human-interpretable strategic meta-cognition. In order to make memory adaptable, we propose a reinforcement-based weight optimization procedure that estimates the empirical utility of each meta-cognition based on reward feedback from downstream tasks. These optimized strategies are then dynamically integrated into the LLM agent's training loop through meta-cognitive prompting. Empirically, the learnable graph memory delivers robust generalization, improves LLM agents' strategic reasoning performance, and provides consistent benefits during Reinforcement Learning (RL) training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06852v2">Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI-26-AIA
    </div>
    <details class="paper-abstract">
      Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11773v4">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents -- virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR. Our code is publicly available at https://github.com/ZikangLeng/AgentSense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07322v2">FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07784v1">Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-agent debate (MAD) has recently emerged as a promising framework for improving the reasoning performance of large language models (LLMs). Yet, whether LLM agents can genuinely engage in deliberative reasoning, beyond simple ensembling or majority voting, remains unclear. We address this question through a controlled study using the Knight--Knave--Spy logic puzzle, which enables precise, step-wise evaluation of debate outcomes and processes under verifiable ground truth. We systematically set up six structural and cognitive factors, including agent team size, composition, confidence visibility, debate order, debate depth, and task difficulty, to disentangle their respective effects on collective reasoning. Our results show that intrinsic reasoning strength and group diversity are the dominant drivers of debate success, while structural parameters such as order or confidence visibility offer limited gains. Beyond outcomes, process-level analyses identify key behavioral patterns: majority pressure suppresses independent correction, effective teams overturn incorrect consensus, and rational, validity-aligned reasoning most strongly predicts improvement. These findings provide valuable insights into how and why LLM debates succeed or fail, offering guidance for designing interpretable and truth-seeking multi-agent reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03764v2">LLM-based Relevance Assessment for Web-Scale Search Evaluation at Pinterest</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 RecSys 2025 EARL Workshop
    </div>
    <details class="paper-abstract">
      Relevance evaluation plays a crucial role in personalized search systems to ensure that search results align with a user's queries and intent. While human annotation is the traditional method for relevance evaluation, its high cost and long turnaround time limit its scalability. In this work, we present our approach at Pinterest Search to automate relevance evaluation for online experiments using fine-tuned LLMs. We rigorously validate the alignment between LLM-generated judgments and human annotations, demonstrating that LLMs can provide reliable relevance measurement for experiments while greatly improving the evaluation efficiency. Leveraging LLM-based labeling further unlocks the opportunities to expand the query set, optimize sampling design, and efficiently assess a wider range of search experiences at scale. This approach leads to higher-quality relevance metrics and significantly reduces the Minimum Detectable Effect (MDE) in online experiment measurements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14429v3">LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 16 pages, 11 figures, Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably stable perplexity during direct context extrapolation. Moreover, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct local perception phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first length extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs. The code is available at https://github.com/OpenMOSS/LongLLaDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01903v2">STAR-1: Safer Alignment of Reasoning LLMs with 1K Data</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      This paper introduces STAR-1, a high-quality, just-1k-scale safety dataset specifically designed for large reasoning models (LRMs) like DeepSeek-R1. Built on three core principles -- diversity, deliberative reasoning, and rigorous filtering -- STAR-1 aims to address the critical needs for safety alignment in LRMs. Specifically, we begin by integrating existing open-source safety datasets from diverse sources. Then, we curate safety policies to generate policy-grounded deliberative reasoning samples. Lastly, we apply a GPT-4o-based safety scoring system to select training examples aligned with best practices. Experimental results show that fine-tuning LRMs with STAR-1 leads to an average 40% improvement in safety performance across four benchmarks, while only incurring a marginal decrease (e.g., an average of 1.1%) in reasoning ability measured across five reasoning tasks. Extensive ablation studies further validate the importance of our design principles in constructing STAR-1 and analyze its efficacy across both LRMs and traditional LLMs. Our project page is https://ucsc-vlaa.github.io/STAR-1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07727v1">LLM-GROP: Visually Grounded Robot Task and Motion Planning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Task planning and motion planning are two of the most important problems in robotics, where task planning methods help robots achieve high-level goals and motion planning methods maintain low-level feasibility. Task and motion planning (TAMP) methods interleave the two processes of task planning and motion planning to ensure goal achievement and motion feasibility. Within the TAMP context, we are concerned with the mobile manipulation (MoMa) of multiple objects, where it is necessary to interleave actions for navigation and manipulation. In particular, we aim to compute where and how each object should be placed given underspecified goals, such as ``set up dinner table with a fork, knife and plate.'' We leverage the rich common sense knowledge from large language models (LLMs), e.g., about how tableware is organized, to facilitate both task-level and motion-level planning. In addition, we use computer vision methods to learn a strategy for selecting base positions to facilitate MoMa behaviors, where the base position corresponds to the robot's ``footprint'' and orientation in its operating space. Altogether, this article provides a principled TAMP framework for MoMa tasks that accounts for common sense about object rearrangement and is adaptive to novel situations that include many objects that need to be moved. We performed quantitative experiments in both real-world settings and simulated environments. We evaluated the success rate and efficiency in completing long-horizon object rearrangement tasks. While the robot completed 84.4\% real-world object rearrangement trials, subjective human evaluations indicated that the robot's performance is still lower than experienced human waiters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07722v1">Critical Confabulation: Can LLMs Hallucinate for Social Good?</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 24 pages, 4 figures, under review
    </div>
    <details class="paper-abstract">
      LLMs hallucinate, yet some confabulations can have social affordances if carefully bounded. We propose critical confabulation (inspired by critical fabulation from literary and social theory), the use of LLM hallucinations to "fill-in-the-gap" for omissions in archives due to social and political inequality, and reconstruct divergent yet evidence-bound narratives for history's "hidden figures". We simulate these gaps with an open-ended narrative cloze task: asking LLMs to generate a masked event in a character-centric timeline sourced from a novel corpus of unpublished texts. We evaluate audited (for data contamination), fully-open models (the OLMo-2 family) and unaudited open-weight and proprietary baselines under a range of prompts designed to elicit controlled and useful hallucinations. Our findings validate LLMs' foundational narrative understanding capabilities to perform critical confabulation, and show how controlled and well-specified hallucinations can support LLM applications for knowledge production without collapsing speculation into a lack of historical accuracy and fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08060v1">From LLMs to Agents: A Comparative Evaluation of LLMs and LLM-based Agents in Security Patch Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      The widespread adoption of open-source software (OSS) has accelerated software innovation but also increased security risks due to the rapid propagation of vulnerabilities and silent patch releases. In recent years, large language models (LLMs) and LLM-based agents have demonstrated remarkable capabilities in various software engineering (SE) tasks, enabling them to effectively address software security challenges such as vulnerability detection. However, systematic evaluation of the capabilities of LLMs and LLM-based agents in security patch detection remains limited. To bridge this gap, we conduct a comprehensive evaluation of the performance of LLMs and LLM-based agents for security patch detection. Specifically, we investigate three methods: Plain LLM (a single LLM with a system prompt), Data-Aug LLM (data augmentation based on the Plain LLM), and the ReAct Agent (leveraging the thought-action-observation mechanism). We also evaluate the performance of both commercial and open-source LLMs under these methods and compare these results with those of existing baselines. Furthermore, we analyze the detection performance of these methods across various vulnerability types, and examine the impact of different prompting strategies and context window sizes on the results. Our findings reveal that the Data-Aug LLM achieves the best overall performance, whereas the ReAct Agent demonstrates the lowest false positive rate (FPR). Although baseline methods exhibit strong accuracy, their false positive rates are significantly higher. In contrast, our evaluated methods achieve comparable accuracy while substantially reducing the FPR. These findings provide valuable insights into the practical applications of LLMs and LLM-based agents in security patch detection, highlighting their advantage in maintaining robust performance while minimizing false positive rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25977v3">NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      AI accelerators, customized to AI workloads, provide cost-effective and high-performance solutions for training and inference. Trainium, an AI accelerator recently developed by Amazon Web Services (AWS), provides an attractive option for LLM training and inference through its heterogeneous architecture. However, leveraging Trainium architecture for high performance can be challenging because of its systolic array architecture and special requirement on data layout. In this paper, we design high-performance matrix multiplication (matmul), a critical compute kernel, for LLM inference on Trainium. We introduce a series of techniques customized to Trainium based on kernel fusion and novel caching strategies to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transpose. Evaluating with nine datasets and four recent LLMs, we show that our system largely outperforms the state-of-the-art matmul implemented by AWS on Trainium: at the level of matmul kernel, it achieves an average 1.35x speedup (up to 2.22x), which translates to an average 1.66x speedup (up to 2.49x) for end-to-end LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25979v3">AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.21239v5">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the Gram matrix determinant of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring internal access to LLMs. We conduct extensive experiments on both external and internal uncertainty detections, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08798v1">Structured Uncertainty guided Clarification for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      LLM agents extend large language models with tool-calling capabilities, but ambiguous user instructions often lead to incorrect invocations and task failures. We introduce a principled formulation of structured uncertainty over tool-call parameters, modeling joint tool-argument clarification as a POMDP with Expected Value of Perfect Information (EVPI) objective for optimal question selection and aspect-based cost modeling to prevent redundancy. Our SAGE-Agent leverages this structured uncertainty to achieve superior efficiency: increasing coverage on ambiguous tasks by 7-39\% while reducing clarification questions by 1.5-2.7$\times$ compared to strong prompting and uncertainty-based baselines. We present ClarifyBench, the first multi-turn tool-augmented disambiguation benchmark with realistic LLM-based user simulation across diverse domains including document editing, vehicle control, and travel booking. Additionally, we demonstrate that structured uncertainty provides effective training signals for reinforcement learning, boosting When2Call accuracy from 36.5\% to 65.2\% (3B model) and 36.7\% to 62.9\% (7B model) through uncertainty-weighted GRPO training. These results establish structured uncertainty as a principled, efficient approach for tool-augmented agents, improving both task success and interaction efficiency in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14884v3">Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 NeurIPS 2025, 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop Selective Head Attention with hardware-efficient, sparsity-aware GPU kernels, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, Qwen, Mistral across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: https://github.com/susavlsh10/Polar-Sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08721v1">Benevolent Dictators? On LLM Agent Behavior in Dictator Games</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 7 pages, 2 figures, v1 init
    </div>
    <details class="paper-abstract">
      In behavioral sciences, experiments such as the ultimatum game are conducted to assess preferences for fairness or self-interest of study participants. In the dictator game, a simplified version of the ultimatum game where only one of two players makes a single decision, the dictator unilaterally decides how to split a fixed sum of money between themselves and the other player. Although recent studies have explored behavioral patterns of AI agents based on Large Language Models (LLMs) instructed to adopt different personas, we question the robustness of these results. In particular, many of these studies overlook the role of the system prompt - the underlying instructions that shape the model's behavior - and do not account for how sensitive results can be to slight changes in prompts. However, a robust baseline is essential when studying highly complex behavioral aspects of LLMs. To overcome previous limitations, we propose the LLM agent behavior study (LLM-ABS) framework to (i) explore how different system prompts influence model behavior, (ii) get more reliable insights into agent preferences by using neutral prompt variations, and (iii) analyze linguistic features in responses to open-ended instructions by LLM agents to better understand the reasoning behind their behavior. We found that agents often exhibit a strong preference for fairness, as well as a significant impact of the system prompt on their behavior. From a linguistic perspective, we identify that models express their responses differently. Although prompt sensitivity remains a persistent challenge, our proposed framework demonstrates a robust foundation for LLM agent behavior studies. Our code artifacts are available at https://github.com/andreaseinwiller/LLM-ABS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08715v1">Bridging Natural Language and ASP: A Hybrid Approach Using LLMs and AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Answer Set Programming (ASP) is a declarative programming paradigm based on logic programming and non-monotonic reasoning. It is a tremendously powerful tool for describing and solving combinatorial problems. Like any other language, ASP requires users to learn how it works and the syntax involved. It is becoming increasingly required for those unfamiliar with programming languages to interact with code. This paper proposes a novel method of translating unconstrained English into ASP programs for logic puzzles using an LLM and Abstract Meaning Representation (AMR) graphs. Everything from ASP rules, facts, and constraints is generated to fully represent and solve the desired problem. Example logic puzzles are used to demonstrate the capabilities of the system. While most current methods rely entirely on an LLM, our system minimizes the role of the LLM only to complete straightforward tasks. The LLM is used to simplify natural language sentences, identify keywords, and generate simple facts. The AMR graphs are then parsed from simplified language and used to generate ASP constraints systematically. The system successfully creates an entire ASP program that solves a combinatorial logic problem. This approach is a significant first step in creating a lighter-weight, explainable system that converts natural language to solve complex logic problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10687v1">Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07568v1">Procedural Knowledge Improves Agentic LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle when performing agentic tasks without substantial tool support, prom-pt engineering, or fine tuning. Despite research showing that domain-dependent, procedural knowledge can dramatically increase planning efficiency, little work evaluates its potential for improving LLM performance on agentic tasks that may require implicit planning. We formalize, implement, and evaluate an agentic LLM workflow that leverages procedural knowledge in the form of a hierarchical task network (HTN). Empirical results of our implementation show that hand-coded HTNs can dramatically improve LLM performance on agentic tasks, and using HTNs can boost a 20b or 70b parameter LLM to outperform a much larger 120b parameter LLM baseline. Furthermore, LLM-created HTNs improve overall performance, though less so. The results suggest that leveraging expertise--from humans, documents, or LLMs--to curate procedural knowledge will become another important tool for improving LLM workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07698v1">Smart but Costly? Benchmarking LLMs on Functional Accuracy and Energy Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of AI technologies and their accelerated adoption in software development necessitates a systematic evaluation of their environmental impact alongside functional correctness. While prior studies have examined sustainability in large language models, existing approaches lack systematic frameworks for evaluating accuracy-energy trade-offs in Code Language Models (CLMs). In this paper, we present a framework, BRACE, to benchmark CLMs on a unified scale of energy efficiency and functional correctness (referred to as accuracy). We benchmark 22 state-of-the-art models on code generation and summarization tasks, proposing two rating methods: Concentric Incremental Rating Circles (CIRC) and Observation to Expectation Rating (OTER). CIRC provides deterministic Euclidean-based rankings with static trade-offs that are robust to outliers, and OTER offers trend-aware evaluation with dynamic trade-offs that capture the complex correlation between energy and accuracy, each offering a distinct perspective and addressing the problem in a unique way. These rating methods enable us to rate LLMs on a 1-5 scale reflecting their combined capabilities in terms of energy efficiency and functional correctness. Our analysis reveals models generally perform better in the code summarization tasks as they are not enforced to generate a grammar-based and syntactically correct output. Also, we find that models' size does not have a significant impact on their ratings, indicating that if models utilize their parameters efficiently, they can be ranked higher on these scales. The proposed BRACE framework empowers practitioners to make evidence-based model selections that balance sustainability with task requirements, guiding rating choice -- CIRC for deterministic comparisons or OTER for trend-aware evaluation -- based on deployment priorities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13697v3">RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based post-training of large language models (LLMs) has recently gained attention, particularly following the release of DeepSeek R1, which applied GRPO for fine-tuning. Amid the growing hype around improved reasoning abilities attributed to RL post-training, we critically examine the formulation and assumptions underlying these methods. We start by highlighting the popular structural assumptions made in modeling LLM training as a Markov Decision Process (MDP), and show how they lead to a degenerate MDP that doesn't quite need the RL/GRPO apparatus. The two critical structural assumptions include (1) making the MDP states be just a concatenation of the actions-with states becoming the context window and the actions becoming the tokens in LLMs and (2) splitting the reward of a state-action trajectory uniformly across the trajectory. Through a comprehensive analysis, we demonstrate that these simplifying assumptions make the approach effectively equivalent to an outcome-driven supervised learning. Our experiments on benchmarks including GSM8K and Countdown using Qwen-2.5 base models show that iterative supervised fine-tuning, incorporating both positive and negative samples, achieves performance comparable to GRPO-based training. We will also argue that the structural assumptions indirectly incentivize the RL to generate longer sequences of intermediate tokens-which in turn feeds into the narrative of "RL generating longer thinking traces." While RL may well be a very useful technique for improving the reasoning abilities of LLMs, our analysis shows that the simplistic structural assumptions made in modeling the underlying MDP render the popular LLM RL frameworks and their interpretations questionable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04715v3">Selection of LLM Fine-Tuning Data based on Orthogonal Rules</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      High-quality training data is critical to the performance of large language models (LLMs). Recent work has explored using LLMs to rate and select data based on a small set of human-designed criteria (rules), but these approaches often rely heavily on heuristics, lack principled metrics for rule evaluation, and generalize poorly to new tasks. We propose a novel rule-based data selection framework that introduces a metric based on the orthogonality of rule score vectors to evaluate and select complementary rules. Our automated pipeline first uses LLMs to generate diverse rules covering multiple aspects of data quality, then rates samples according to these rules and applies the determinantal point process (DPP) to select the most independent rules. These rules are then used to score the full dataset, and high-scoring samples are selected for downstream tasks such as LLM fine-tuning. We evaluate our framework in two experiment setups: (1) alignment with ground-truth ratings and (2) performance of LLMs fine-tuned on the selected data. Experiments across IMDB, Medical, Math, and Code domains demonstrate that our DPP-based rule selection consistently improves both rating accuracy and downstream model performance over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07669v1">Making LLMs Reliable When It Matters Most: A Five-Layer Architecture for High-Stakes Decisions</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 24 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) excel in verifiable domains where outputs can be checked before action but prove less reliable for high-stakes strategic decisions with uncertain outcomes. This gap, driven by mutually reinforcing cognitive biases in both humans and artificial intelligence (AI) systems, threatens the defensibility of valuations and sustainability of investments in the sector. This report describes a framework emerging from systematic qualitative assessment across 7 frontier-grade LLMs and 3 market-facing venture vignettes under time pressure. Detailed prompting specifying decision partnership and explicitly instructing avoidance of sycophancy, confabulation, solution drift, and nihilism achieved initial partnership state but failed to maintain it under operational pressure. Sustaining protective partnership state required an emergent 7-stage calibration sequence, built upon a 4-stage initialization process, within a 5-layer protection architecture enabling bias self-monitoring, human-AI adversarial challenge, partnership state verification, performance degradation detection, and stakeholder protection. Three discoveries resulted: partnership state is achievable through ordered calibration but requires emergent maintenance protocols; reliability degrades when architectural drift and context exhaustion align; and dissolution discipline prevents costly pursuit of fundamentally wrong directions. Cross-model validation revealed systematic performance differences across LLM architectures. This approach demonstrates that human-AI teams can achieve cognitive partnership capable of preventing avoidable regret in high-stakes decisions, addressing return-on-investment expectations that depend on AI systems supporting consequential decision-making without introducing preventable cognitive traps when verification arrives too late.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07659v1">Revisiting NLI: Towards Cost-Effective and Human-Aligned Metrics for Evaluating LLMs in Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Evaluating answers from state-of-the-art large language models (LLMs) is challenging: lexical metrics miss semantic nuances, whereas "LLM-as-Judge" scoring is computationally expensive. We re-evaluate a lightweight alternative -- off-the-shelf Natural Language Inference (NLI) scoring augmented by a simple lexical-match flag and find that this decades-old technique matches GPT-4o's accuracy (89.9%) on long-form QA, while requiring orders-of-magnitude fewer parameters. To test human alignment of these metrics rigorously, we introduce DIVER-QA, a new 3000-sample human-annotated benchmark spanning five QA datasets and five candidate LLMs. Our results highlight that inexpensive NLI-based evaluation remains competitive and offer DIVER-QA as an open resource for future metric research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09205v4">Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at EUSIPCO 2025 - 5 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07641v1">LLMs vs. Traditional Sentiment Tools in Psychology: An Evaluation on Belgian-Dutch Narratives</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Understanding emotional nuances in everyday language is crucial for computational linguistics and emotion research. While traditional lexicon-based tools like LIWC and Pattern have served as foundational instruments, Large Language Models (LLMs) promise enhanced context understanding. We evaluated three Dutch-specific LLMs (ChocoLlama-8B-Instruct, Reynaerde-7B-chat, and GEITje-7B-ultra) against LIWC and Pattern for valence prediction in Flemish, a low-resource language variant. Our dataset comprised approximately 25000 spontaneous textual responses from 102 Dutch-speaking participants, each providing narratives about their current experiences with self-assessed valence ratings (-50 to +50). Surprisingly, despite architectural advancements, the Dutch-tuned LLMs underperformed compared to traditional methods, with Pattern showing superior performance. These findings challenge assumptions about LLM superiority in sentiment analysis tasks and highlight the complexity of capturing emotional valence in spontaneous, real-world narratives. Our results underscore the need for developing culturally and linguistically tailored evaluation frameworks for low-resource language variants, while questioning whether current LLM fine-tuning approaches adequately address the nuanced emotional expressions found in everyday language use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07637v1">Private-RAG: Answering Multiple Queries with LLMs while Keeping Your Data Private</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving documents from an external corpus at inference time. When this corpus contains sensitive information, however, unprotected RAG systems are at risk of leaking private information. Prior work has introduced differential privacy (DP) guarantees for RAG, but only in single-query settings, which fall short of realistic usage. In this paper, we study the more practical multi-query setting and propose two DP-RAG algorithms. The first, MURAG, leverages an individual privacy filter so that the accumulated privacy loss only depends on how frequently each document is retrieved rather than the total number of queries. The second, MURAG-ADA, further improves utility by privately releasing query-specific thresholds, enabling more precise selection of relevant documents. Our experiments across multiple LLMs and datasets demonstrate that the proposed methods scale to hundreds of queries within a practical DP budget ($\varepsilon\approx10$), while preserving meaningful utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07628v1">Beyond Correctness: Evaluating and Improving LLM Feedback in Statistical Education</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been proposed as scalable tools to address the gap between the importance of individualized written feedback and the practical challenges of providing it at scale. However, concerns persist regarding the accuracy, depth, and pedagogical value of their feedback responses. The present study investigates the extent to which LLMs can generate feedback that aligns with educational theory and compares techniques to improve their performance. Using mock in-class exam data from two consecutive years of an introductory statistics course at LMU Munich, we evaluated GPT-generated feedback against an established but expanded pedagogical framework. Four enhancement methods were compared in a highly standardized setting, making meaningful comparisons possible: Using a state-of-the-art model, zero-shot prompting, few-shot prompting, and supervised fine-tuning using Low-Rank Adaptation (LoRA). Results show that while all LLM setups reliably provided correctness judgments and explanations, their ability to deliver contextual feedback and suggestions on how students can monitor and regulate their own learning remained limited. Among the tested methods, zero-shot prompting achieved the strongest balance between quality and cost, while fine-tuning required substantially more resources without yielding clear advantages. For educators, this suggests that carefully designed prompts can substantially improve the usefulness of LLM feedback, making it a promising tool, particularly in large introductory courses where students would otherwise receive little or no written feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17120v2">Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      We have only limited understanding of how and why large language models (LLMs) respond in the ways that they do. Their neural networks have proven challenging to interpret, and we are only beginning to tease out the function of individual neurons and circuits within them. However, another path to understanding these systems is to investigate and develop their capacity to explain their own functioning. Here, we show that i) LLMs can accurately describe quantitative features of their own internal processes during certain kinds of decision-making and ii) that it is possible to improve these capabilities through training. To do so, we fine-tuned GPT-4o and GPT-4o-mini to make decisions in a wide variety of complex contexts (e.g., choosing between condos, loans, vacations, etc.) according to randomly-generated, quantitative preferences about how to weigh different attributes (e.g., the relative importance of natural light versus quiet surroundings for condos). We demonstrate that the LLMs can accurately report these preferences (i.e., the weights that they learned to give to different attributes during decision-making). Next, we demonstrate that these LLMs can be fine-tuned to explain their decision-making even more accurately. Finally, we demonstrate that this training generalizes: It improves the ability of the models to accurately explain how they make other complex decisions, not just decisions they have been fine-tuned to make. This work is a step towards training LLMs to accurately and broadly report on their own internal processes -- a possibility that would yield substantial benefits for interpretability, control, and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07585v1">LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 11 pages, 5 figures. To appear in AI4F @ ACM ICAIF '25, November 15-18, 2025, Singapore
    </div>
    <details class="paper-abstract">
      Financial institutions deploy Large Language Models (LLMs) for reconciliations, regulatory reporting, and client communications, but nondeterministic outputs (output drift) undermine auditability and trust. We quantify drift across five model architectures (7B-120B parameters) on regulated financial tasks, revealing a stark inverse relationship: smaller models (Granite-3-8B, Qwen2.5-7B) achieve 100% output consistency at T=0.0, while GPT-OSS-120B exhibits only 12.5% consistency (95% CI: 3.5-36.0%) regardless of configuration (p<0.0001, Fisher's exact test). This finding challenges conventional assumptions that larger models are universally superior for production deployment. Our contributions include: (i) a finance-calibrated deterministic test harness combining greedy decoding (T=0.0), fixed seeds, and SEC 10-K structure-aware retrieval ordering; (ii) task-specific invariant checking for RAG, JSON, and SQL outputs using finance-calibrated materiality thresholds (plus or minus 5%) and SEC citation validation; (iii) a three-tier model classification system enabling risk-appropriate deployment decisions; and (iv) an audit-ready attestation system with dual-provider validation. We evaluated five models (Qwen2.5-7B via Ollama, Granite-3-8B via IBM watsonx.ai, Llama-3.3-70B, Mistral-Medium-2505, and GPT-OSS-120B) across three regulated financial tasks. Across 480 runs (n=16 per condition), structured tasks (SQL) remain stable even at T=0.2, while RAG tasks show drift (25-75%), revealing task-dependent sensitivity. Cross-provider validation confirms deterministic behavior transfers between local and cloud deployments. We map our framework to Financial Stability Board (FSB), Bank for International Settlements (BIS), and Commodity Futures Trading Commission (CFTC) requirements, demonstrating practical pathways for compliance-ready AI deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.09235v3">Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form QA</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Proceedings of the 9th Widening NLP Workshop
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) as chat assistants capable of generating human-like conversations has amplified the need for robust evaluation methods, particularly for open-ended tasks. Conventional metrics such as EM and F1, while useful, are inadequate for capturing the full semantics and contextual depth of such generative outputs. We propose a reference-guided verdict method that automates the evaluation process by leveraging multiple LLMs as judges. Through experiments on free-form question-answering tasks, we demonstrate that combining multiple models improves the reliability and accuracy of evaluations, especially in tasks where a single model may struggle. The results indicate a strong correlation with human evaluations, establishing the proposed method as a reliable alternative to traditional metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20166v2">CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08542v2">CLEV: LLM-Based Evaluation Through Lightweight Efficient Voting for Free-Form Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to AACL 2025
    </div>
    <details class="paper-abstract">
      Evaluating free-form Question Answering (QA) remains a challenge due to its diverse and open-ended nature. Traditional automatic metrics fail to capture semantic equivalence or accommodate the variability of open-ended responses. Leveraging Large Language Models (LLMs) as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. We propose Consensus via Lightweight Efficient Voting (CLEV), which employs two primary LLMs as judges and invokes a third judge only in cases of disagreement. This approach prioritizes evaluation reliability while reducing unnecessary computational demands. Through experiments, including human evaluation, we demonstrate CLEV's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating LLMs on free-form QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07555v1">LLM Optimization Unlocks Real-Time Pairwise Reranking</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Efficiently reranking documents retrieved from information retrieval (IR) pipelines to enhance overall quality of Retrieval-Augmented Generation (RAG) system remains an important yet challenging problem. Recent studies have highlighted the importance of Large Language Models (LLMs) in reranking tasks. In particular, Pairwise Reranking Prompting (PRP) has emerged as a promising plug-and-play approach due to its usability and effectiveness. However, the inherent complexity of the algorithm, coupled with the high computational demands and latency incurred due to LLMs, raises concerns about its feasibility in real-time applications. To address these challenges, this paper presents a focused study on pairwise reranking, demonstrating that carefully applied optimization methods can significantly mitigate these issues. By implementing these methods, we achieve a remarkable latency reduction of up to 166 times, from 61.36 seconds to 0.37 seconds per query, with an insignificant drop in performance measured by Recall@k. Our study highlights the importance of design choices that were previously overlooked, such as using smaller models, limiting the reranked set, using lower precision, reducing positional bias with one-directional order inference, and restricting output tokens. These optimizations make LLM-based reranking substantially more efficient and feasible for latency-sensitive, real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07364v1">Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07318v1">When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01934v2">Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 EMNLP 2025 finding
    </div>
    <details class="paper-abstract">
      Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02735v2">Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Prompt injection attack has been listed as the top-1 security threat to LLM-integrated applications, which interact with external environment data for complex tasks. The untrusted data may contain an injected prompt trying to arbitrarily manipulate the system. Model-level prompt injection defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source secure models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigating prompt injection attacks. To this end, we develop Meta SecAlign, the first fully open-source LLM with built-in model-level defense that achieves commercial-grade performance, powerful enough for complex agentic tasks. We provide complete details of our training recipe, an improved version of the SOTA SecAlign defense. We perform the most comprehensive evaluation to date on 9 utility benchmarks and 7 security benchmarks on general knowledge, instruction following, and agentic workflows. Results show that Meta SecAlign, despite being trained on generic instruction-tuning samples only, surprisingly confers security in unseen downstream tasks, including tool-calling and web-navigation, in addition to general instruction-following. Our best model -- Meta-SecAlign-70B -- establishes a new frontier of utility-security trade-off for open-source LLMs. Even compared to closed-course commercial models such as GPT-5, our model is much securer than most of them. Below are links for the code (https://github.com/facebookresearch/Meta_SecAlign), Meta-SecAlign-70B(https://huggingface.co/facebook/Meta-SecAlign-70B), and Meta-SecAlign-8B(https://huggingface.co/facebook/Meta-SecAlign-8B) models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.13677v3">HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety. The code and dataset are available at https://github.com/wooozihui/HumorReject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08736v2">ChemBOMAS: Accelerated BO in Chemistry with LLM-Enhanced Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Bayesian optimization (BO) is a powerful tool for scientific discovery in chemistry, yet its efficiency is often hampered by the sparse experimental data and vast search space. Here, we introduce ChemBOMAS: a large language model (LLM)-enhanced multi-agent system that accelerates BO through synergistic data- and knowledge-driven strategies. Firstly, the data-driven strategy involves an 8B-scale LLM regressor fine-tuned on a mere 1% labeled samples for pseudo-data generation, robustly initializing the optimization process. Secondly, the knowledge-driven strategy employs a hybrid Retrieval-Augmented Generation approach to guide LLM in dividing the search space while mitigating LLM hallucinations. An Upper Confidence Bound algorithm then identifies high-potential subspaces within this established partition. Across the LLM-refined subspaces and supported by LLM-generated data, BO achieves the improvement of effectiveness and efficiency. Comprehensive evaluations across multiple scientific benchmarks demonstrate that ChemBOMAS set a new state-of-the-art, accelerating optimization efficiency by up to 5-fold compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07229v1">LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 4 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This paper introduces LLMServingSim2.0, a system simulator designed for exploring heterogeneous hardware in large-scale LLM serving systems. LLMServingSim2.0 addresses two key limitations of its predecessor: (1) integrating hardware models into system-level simulators is non-trivial due to the lack of a clear abstraction, and (2) existing simulators support only a narrow subset of serving techniques, leaving no infrastructure that captures the breadth of approaches in modern LLM serving. To overcome these issues, LLMServingSim2.0 adopts trace-driven performance modeling, accompanied by an operator-level latency profiler, enabling the integration of new accelerators with a single command. It further embeds up-to-date serving techniques while exposing flexible interfaces for request routing, cache management, and scheduling policies. In a TPU case study, our profiler requires 18.5x fewer LoC and outperforms the predecessor's hardware-simulator integration, demonstrating LLMServingSim2.0's low-effort hardware extensibility. Our experiments further show that LLMServingSim2.0 reproduces GPU-based LLM serving with 1.9% error, while maintaining practical simulation time, making it a comprehensive platform for both hardware developers and LLM service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07223v1">NoteEx: Interactive Visual Context Manipulation for LLM-Assisted Exploratory Data Analysis in Computational Notebooks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Computational notebooks have become popular for Exploratory Data Analysis (EDA), augmented by LLM-based code generation and result interpretation. Effective LLM assistance hinges on selecting informative context -- the minimal set of cells whose code, data, or outputs suffice to answer a prompt. As notebooks grow long and messy, users can lose track of the mental model of their analysis. They thus fail to curate appropriate contexts for LLM tasks, causing frustration and tedious prompt engineering. We conducted a formative study (n=6) that surfaced challenges in LLM context selection and mental model maintenance. Therefore, we introduce NoteEx, a JupyterLab extension that provides a semantic visualization of the EDA workflow, allowing analysts to externalize their mental model, specify analysis dependencies, and enable interactive selection of task-relevant contexts for LLMs. A user study (n=12) against a baseline shows that NoteEx improved mental model retention and context selection, leading to more accurate and relevant LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07204v1">Evaluating Online Moderation Via LLM-Powered Counterfactual Simulations</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted for publication at AAAI Conference on Artificial Intelligence 2026
    </div>
    <details class="paper-abstract">
      Online Social Networks (OSNs) widely adopt content moderation to mitigate the spread of abusive and toxic discourse. Nonetheless, the real effectiveness of moderation interventions remains unclear due to the high cost of data collection and limited experimental control. The latest developments in Natural Language Processing pave the way for a new evaluation approach. Large Language Models (LLMs) can be successfully leveraged to enhance Agent-Based Modeling and simulate human-like social behavior with unprecedented degree of believability. Yet, existing tools do not support simulation-based evaluation of moderation strategies. We fill this gap by designing a LLM-powered simulator of OSN conversations enabling a parallel, counterfactual simulation where toxic behavior is influenced by moderation interventions, keeping all else equal. We conduct extensive experiments, unveiling the psychological realism of OSN agents, the emergence of social contagion phenomena and the superior effectiveness of personalized moderation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.07863v3">Robust Hallucination Detection in LLMs via Adaptive Token Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) pose significant safety concerns that impede their broader deployment. Recent research in hallucination detection has demonstrated that LLMs' internal representations contain truthfulness hints, which can be harnessed for detector training. However, the performance of these detectors is heavily dependent on the internal representations of predetermined tokens, fluctuating considerably when working on free-form generations with varying lengths and sparse distributions of hallucinated entities. To address this, we propose HaMI, a novel approach that enables robust detection of hallucinations through adaptive selection and learning of critical tokens that are most indicative of hallucinations. We achieve this robustness by an innovative formulation of the Hallucination detection task as Multiple Instance (HaMI) learning over token-level representations within a sequence, thereby facilitating a joint optimisation of token selection and hallucination detection on generation sequences of diverse forms. Comprehensive experimental results on four hallucination benchmarks show that HaMI significantly outperforms existing state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07168v1">LEAD: LLM-enhanced Engine for Author Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Author Name Disambiguation (AND) is a long-standing challenge in bibliometrics and scientometrics, as name ambiguity undermines the accuracy of bibliographic databases and the reliability of research evaluation. This study addresses the problem of cross-source disambiguation by linking academic career records from CercaUniversità, the official registry of Italian academics, with author profiles in Scopus. We introduce LEAD (LLM-enhanced Engine for Author Disambiguation), a novel hybrid framework that combines semantic features extracted through Large Language Models (LLMs) with structural evidence derived from co-authorship and citation networks. Using a gold standard of 606 ambiguous cases, we compare five methods: (i) Label Spreading on co-authorship networks; (ii) Bibliographic Coupling on citation networks; (iii) a standalone LLM-based approach; (iv) an LLM-enriched configuration; and (v) the proposed hybrid pipeline. LEAD achieves the best performance (F1 = 96.7%, accuracy = 95.7%) with lower computational cost than full LLM models. Bibliographic Coupling emerges as the fastest and strongest single-source method. These findings demonstrate that integrating semantic and structural signals within a selective hybrid strategy offers a robust and scalable solution to cross-database author identification. Beyond the Italian case, this work highlights the potential of hybrid LLM-based methods to improve data quality and reliability in scientometric analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07166v1">AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19670v2">CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 19 pages,8 figures
    </div>
    <details class="paper-abstract">
      We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07126v1">Saliency Map-Guided Knowledge Discovery for Subclass Identification with LLM-Based Symbolic Approximations</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      This paper proposes a novel neuro-symbolic approach for sensor signal-based knowledge discovery, focusing on identifying latent subclasses in time series classification tasks. The approach leverages gradient-based saliency maps derived from trained neural networks to guide the discovery process. Multiclass time series classification problems are transformed into binary classification problems through label subsumption, and classifiers are trained for each of these to yield saliency maps. The input signals, grouped by predicted class, are clustered under three distinct configurations. The centroids of the final set of clusters are provided as input to an LLM for symbolic approximation and fuzzy knowledge graph matching to discover the underlying subclasses of the original multiclass problem. Experimental results on well-established time series classification datasets demonstrate the effectiveness of our saliency map-driven method for knowledge discovery, outperforming signal-only baselines in both clustering and subclass identification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07107v1">MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07099v1">E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at https://wxzyd123.github.io/e2e-vguard/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07086v1">LLM Driven Processes to Foster Explainable AI</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      We present a modular, explainable LLM-agent pipeline for decision support that externalizes reasoning into auditable artifacts. The system instantiates three frameworks: Vester's Sensitivity Model (factor set, signed impact matrix, systemic roles, feedback loops); normal-form games (strategies, payoff matrix, equilibria); and sequential games (role-conditioned agents, tree construction, backward induction), with swappable modules at every step. LLM components (default: GPT-5) are paired with deterministic analyzers for equilibria and matrix-based role classification, yielding traceable intermediates rather than opaque outputs. In a real-world logistics case (100 runs), mean factor alignment with a human baseline was 55.5\% over 26 factors and 62.9\% on the transport-core subset; role agreement over matches was 57\%. An LLM judge using an eight-criterion rubric (max 100) scored runs on par with a reconstructed human baseline. Configurable LLM pipelines can thus mimic expert workflows with transparent, inspectable steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07083v1">Increasing AI Explainability by LLM Driven Standard Processes</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      This paper introduces an approach to increasing the explainability of artificial intelligence (AI) systems by embedding Large Language Models (LLMs) within standardized analytical processes. While traditional explainable AI (XAI) methods focus on feature attribution or post-hoc interpretation, the proposed framework integrates LLMs into defined decision models such as Question-Option-Criteria (QOC), Sensitivity Analysis, Game Theory, and Risk Management. By situating LLM reasoning within these formal structures, the approach transforms opaque inference into transparent and auditable decision traces. A layered architecture is presented that separates the reasoning space of the LLM from the explainable process space above it. Empirical evaluations show that the system can reproduce human-level decision logic in decentralized governance, systems analysis, and strategic reasoning contexts. The results suggest that LLM-driven standard processes provide a foundation for reliable, interpretable, and verifiable AI-supported decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07074v1">Importance-Aware Data Selection for Efficient LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted by AAAI 2026 Oral
    </div>
    <details class="paper-abstract">
      Instruction tuning plays a critical role in enhancing the performance and efficiency of Large Language Models (LLMs). Its success depends not only on the quality of the instruction data but also on the inherent capabilities of the LLM itself. Some studies suggest that even a small amount of high-quality data can achieve instruction fine-tuning results that are on par with, or even exceed, those from using a full-scale dataset. However, rather than focusing solely on calculating data quality scores to evaluate instruction data, there is a growing need to select high-quality data that maximally enhances the performance of instruction tuning for a given LLM. In this paper, we propose the Model Instruction Weakness Value (MIWV) as a novel metric to quantify the importance of instruction data in enhancing model's capabilities. The MIWV metric is derived from the discrepancies in the model's responses when using In-Context Learning (ICL), helping identify the most beneficial data for enhancing instruction tuning performance. Our experimental results demonstrate that selecting only the top 1\% of data based on MIWV can outperform training on the full dataset. Furthermore, this approach extends beyond existing research that focuses on data quality scoring for data selection, offering strong empirical evidence supporting the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07070v1">RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervised fine-tuning (SFT) can specialize models but often triggers a ``seesaw'' between in-distribution gains and out-of-distribution robustness, especially for smaller models. To address these challenges, we introduce RedOne 2.0, an SNS-oriented LLM trained with a progressive, RL-prioritized post-training paradigm designed for rapid and stable adaptation. The pipeline consist in three stages: (1) Exploratory Learning on curated SNS corpora to establish initial alignment and identify systematic weaknesses; (2) Targeted Fine-Tuning that selectively applies SFT to the diagnosed gaps while mixing a small fraction of general data to mitigate forgetting; and (3) Refinement Learning that re-applies RL with SNS-centric signals to consolidate improvements and harmonize trade-offs across tasks. Across various tasks spanning three categories, our 4B scale model delivers an average improvements about 2.41 over the 7B sub-optimal baseline. Additionally, RedOne 2.0 achieves average performance lift about 8.74 from the base model with less than half the data required by SFT-centric method RedOne, evidencing superior data efficiency and stability at compact scales. Overall, RedOne 2.0 establishes a competitive, cost-effective baseline for domain-specific LLMs in SNS scenario, advancing capability without sacrificing robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07044v1">Evaluating LLMs for Anxiety, Depression, and Stress Detection Evaluating Large Language Models for Anxiety, Depression, and Stress Detection: Insights into Prompting Strategies and Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Mental health disorders affect over one-fifth of adults globally, yet detecting such conditions from text remains challenging due to the subtle and varied nature of symptom expression. This study evaluates multiple approaches for mental health detection, comparing Large Language Models (LLMs) such as Llama and GPT with classical machine learning and transformer-based architectures including BERT, XLNet, and Distil-RoBERTa. Using the DAIC-WOZ dataset of clinical interviews, we fine-tuned models for anxiety, depression, and stress classification and applied synthetic data generation to mitigate class imbalance. Results show that Distil-RoBERTa achieved the highest F1 score (0.883) for GAD-2, while XLNet outperformed others on PHQ tasks (F1 up to 0.891). For stress detection, a zero-shot synthetic approach (SD+Zero-Shot-Basic) reached an F1 of 0.884 and ROC AUC of 0.886. Findings demonstrate the effectiveness of transformer-based models and highlight the value of synthetic data in improving recall and generalization. However, careful calibration is required to prevent precision loss. Overall, this work emphasizes the potential of combining advanced language models and data augmentation to enhance automated mental health assessment from text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07035v1">GoCkpt: Gradient-Assisted Multi-Step overlapped Checkpointing for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 12 pages, 10 figures, 1 tables
    </div>
    <details class="paper-abstract">
      The accuracy of large language models (LLMs) improves with increasing model size, but increasing model complexity also poses significant challenges to training stability. Periodic checkpointing is a key mechanism for fault recovery and is widely used in LLM training. However, traditional checkpointing strategies often pause or delay GPU computation during checkpoint saving for checkpoint GPU-CPU transfer, resulting in significant training interruptions and reduced training throughput. To address this issue, we propose GoCkpt, a method to overlap checkpoint saving with multiple training steps and restore the final checkpoint on the CPU. We transfer the checkpoint across multiple steps, each step transfers part of the checkpoint state, and we transfer some of the gradient data used for parameter updates. After the transfer is complete, each partial checkpoint state is updated to a consistent version on the CPU, thus avoiding the checkpoint state inconsistency problem caused by transferring checkpoints across multiple steps. Furthermore, we introduce a transfer optimization strategy to maximize GPU-CPU bandwidth utilization and SSD persistence throughput. This dual optimization overlapping saves across steps and maximizing I/O efficiency significantly reduces invalid training time. Experimental results show that GoCkpt can increase training throughput by up to 38.4% compared to traditional asynchronous checkpoint solutions in the industry. We also find that GoCkpt can reduce training interruption time by 86.7% compared to the state-of-the-art checkpoint transfer methods, which results in a 4.8% throughput improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14496v2">LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 This is the author's version of a paper submitted to IEEE Intelligent Systems. 2 Tables, 2 Figures
    </div>
    <details class="paper-abstract">
      Swarm intelligence describes how simple, decentralized agents can collectively produce complex behaviors. Recently, the concept of swarming has been extended to large language model (LLM)-powered systems, such as OpenAI's Swarm (OAS) framework, where agents coordinate through natural language prompts. This paper evaluates whether such systems capture the fundamental principles of classical swarm intelligence: decentralization, simplicity, emergence, and scalability. Using OAS, we implement and compare classical and LLM-based versions of two well-established swarm algorithms: Boids and Ant Colony Optimization. Results indicate that while LLM-powered swarms can emulate swarm-like dynamics, they are constrained by substantial computational overhead. For instance, our LLM-based Boids simulation required roughly 300x more computation time than its classical counterpart, highlighting current limitations in applying LLM-driven swarms to real-time systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07033v1">Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Paper has been accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable, concerns over the unauthorized use of copyrighted and licensed content in their training data have grown, especially in the context of code. Open-source code, often protected by open source licenses (e.g, GPL), poses legal and ethical challenges when used in pretraining. Detecting whether specific code samples were included in LLM training data is thus critical for transparency, accountability, and copyright compliance. We propose SynPrune, a syntax-pruned membership inference attack method tailored for code. Unlike prior MIA approaches that treat code as plain text, SynPrune leverages the structured and rule-governed nature of programming languages. Specifically, it identifies and excludes consequent tokens that are syntactically required and not reflective of authorship, from attribution when computing membership scores. Experimental results show that SynPrune consistently outperforms the state-of-the-arts. Our method is also robust across varying function lengths and syntax categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07017v1">Benchmarking LLMs for Fine-Grained Code Review with Enriched Context in Practice</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Code review is a cornerstone of software quality assurance, and recent advances in Large Language Models (LLMs) have shown promise in automating this process. However, existing benchmarks for LLM-based code review face three major limitations. (1) Lack of semantic context: most benchmarks provide only code diffs without textual information such as issue descriptions, which are crucial for understanding developer intent. (2) Data quality issues: without rigorous validation, many samples are noisy-e.g., reviews on outdated or irrelevant code-reducing evaluation reliability. (3) Coarse granularity: most benchmarks operate at the file or commit level, overlooking the fine-grained, line-level reasoning essential for precise review. We introduce ContextCRBench, a high-quality, context-rich benchmark for fine-grained LLM evaluation in code review. Our construction pipeline comprises: (1) Raw Data Crawling, collecting 153.7K issues and pull requests from top-tier repositories; (2) Comprehensive Context Extraction, linking issue-PR pairs for textual context and extracting the full surrounding function or class for code context; and (3) Multi-stage Data Filtering, combining rule-based and LLM-based validation to remove outdated, malformed, or low-value samples, resulting in 67,910 context-enriched entries. ContextCRBench supports three evaluation scenarios aligned with the review workflow: (1) hunk-level quality assessment, (2) line-level defect localization, and (3) line-level comment generation. Evaluating eight leading LLMs (four closed-source and four open-source) reveals that textual context yields greater performance gains than code context alone, while current LLMs remain far from human-level review ability. Deployed at ByteDance, ContextCRBench drives a self-evolving code review system, improving performance by 61.98% and demonstrating its robustness and industrial utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17631v3">Time-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting, but this progress is still met with skepticism about whether LLMs are truly useful for this task. To address this, we propose Time-Prompt, a framework for activating LLMs for time series forecasting. Specifically, we first construct a unified prompt paradigm with learnable soft prompts to guide the LLM's behavior and textualized hard prompts to enhance the time series representations. Second, to enhance LLM' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve fusion of temporal and textual data. Finally, we efficiently fine-tune the LLM's parameters using time series data. Furthermore, we focus on carbon emissions, aiming to provide a modest contribution to global carbon neutrality. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that Time-Prompt is a powerful framework for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24826v2">LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 10 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework. Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: https://github.com/lyxx3rd/LegalEval-Q.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16040v2">Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 23 pages, Published in Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent advances in test-time scaling of large language models (LLMs), exemplified by DeepSeek-R1 and OpenAI's o1, show that extending the chain of thought during inference can significantly improve general reasoning performance. However, the impact of this paradigm on legal reasoning remains insufficiently explored. To address this gap, we present the first systematic evaluation of 12 LLMs, including both reasoning-focused and general-purpose models, across 17 Chinese and English legal tasks spanning statutory and case-law traditions. In addition, we curate a bilingual chain-of-thought dataset for legal reasoning through distillation from DeepSeek-R1 and develop Legal-R1, an open-source model specialized for the legal domain. Experimental results show that Legal-R1 delivers competitive performance across diverse tasks. DeepSeek-R1 exhibits clear advantages in Chinese legal reasoning, while OpenAI's o1 achieves comparable results on English tasks. We further conduct a detailed error analysis, which reveals recurring issues such as outdated legal knowledge, limited capacity for legal interpretation, and susceptibility to factual hallucinations. These findings delineate the main obstacles confronting legal-domain LLMs and suggest promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06890v1">EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 22 pages, 9 figures, accepted by AAAI2026 as oral paper
    </div>
    <details class="paper-abstract">
      Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See https://github.com/YL1N/EduGuardBench for Materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10027v3">LLMCARE: early detection of cognitive impairment via transformer models enhanced by LLM-generated synthetic data</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Alzheimer's disease and related dementias(ADRD) affect nearly five million older adults in the United States, yet more than half remain undiagnosed. Speech-based natural language processing(NLP) offers a scalable approach for detecting early cognitive decline through subtle linguistic markers that may precede clinical diagnosis. This study develops and evaluates a speech-based screening pipeline integrating transformer embeddings with handcrafted linguistic features, synthetic augmentation using large language models(LLMs), and benchmarking of unimodal and multimodal classifiers. External validation assessed generalizability to a MCI-only cohort. Transcripts were drawn from the ADReSSo 2021 benchmark dataset(n=237, Pitt Corpus) and the DementiaBank Delaware corpus(n=205, MCI vs. controls). Ten transformer models were tested under three fine-tuning strategies. A late-fusion model combined embeddings from the top transformer with 110 linguistic features. Five LLMs(LLaMA8B/70B, MedAlpaca7B, Ministral8B,GPT-4o) generated label-conditioned synthetic speech for augmentation, and three multimodal LLMs(GPT-4o,Qwen-Omni,Phi-4) were evaluated in zero-shot and fine-tuned modes. On ADReSSo, the fusion model achieved F1=83.3(AUC=89.5), outperforming transformer-only and linguistic baselines. MedAlpaca7B augmentation(2x) improved F1=85.7, though larger scales reduced gains. Fine-tuning boosted unimodal LLMs(MedAlpaca7B F1=47.7=>78.7), while multimodal models performed lower (Phi-4=71.6;GPT-4o=67.6). On Delaware, the fusion plus 1x MedAlpaca7B model achieved F1=72.8(AUC=69.6). Integrating transformer and linguistic features enhances ADRD detection. LLM-based augmentation improves data efficiency but yields diminishing returns, while current multimodal models remain limited. Validation on an independent MCI cohort supports the pipeline's potential for scalable, clinically relevant early screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.19638v2">LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 This work has been accepted and published in the IEEE Access journal. This arXiv version is retained for archival purposes. Readers should use and cite the IEEE Access Version available at https://ieeexplore.ieee.org/document/10900365
    </div>
    <details class="paper-abstract">
      With the ever-increasing number of news stories available online, classifying them by topic, regardless of the language they are written in, has become crucial for enhancing readers' access to relevant content. To address this challenge, we propose a teacher-student framework based on large language models (LLMs) for developing multilingual news topic classification models of reasonable size with no need for manual data annotation. The framework employs a Generative Pretrained Transformer (GPT) model as the teacher model to develop a news topic training dataset through automatic annotation of 20,000 news articles in Slovenian, Croatian, Greek, and Catalan. Articles are classified into 17 main categories from the Media Topic schema, developed by the International Press Telecommunications Council (IPTC). The teacher model exhibits high zero-shot performance in all four languages. Its agreement with human annotators is comparable to that between the human annotators themselves. To mitigate the computational limitations associated with the requirement of processing millions of texts daily, smaller BERT-like student models are fine-tuned on the GPT-annotated dataset. These student models achieve high performance comparable to the teacher model. Furthermore, we explore the impact of the training data size on the performance of the student models and investigate their monolingual, multilingual, and zero-shot cross-lingual capabilities. The findings indicate that student models can achieve high performance with a relatively small number of training instances, and demonstrate strong zero-shot cross-lingual abilities. Finally, we publish the best-performing news topic classifier, enabling multilingual classification with the top-level categories of the IPTC Media Topic schema.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.00708v3">DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13557v5">MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Coarse-grained Reconfigurable Arrays (CGRAs) are a promising computing architecture that can deliver high-performance, energy-efficient acceleration across diverse domains. By supporting reconfiguration at the functional unit level, CGRAs efficiently adapt to varying computational patterns and optimize resource utilization. However, designing CGRAs is highly challenging due to the vast design space, independent architectural parameters, and the time-consuming nature of manual design. Fortunately, the rapid advancement of large language models (LLMs) presents new opportunities to automate this process. In this work, we propose MACO-- an open-source multi-agent LLM-based framework for Hardware/Software (HW/SW) co-design of CGRAs. The framework employs LLM reasoning to generate CGRAs across four stages: HW/SW co-design, Design error correction, Best design selection, and Evaluation & Feedback. Furthermore, MACO iteratively optimizes the generated CGRAs, leveraging agent reasoning and feedback to achieve higher PPA (that is, power, performance, and area) design points for a given domain. In addition, we introduce an LLM self-learning mechanism that employs LLM-driven decision making to select the optimal CGRA to accelerate the design process. We evaluate the framework with state-of-the-art LLM-based methods and manual CGRA design, in terms of performance, power consumption, and area. Experimental results show that MACO efficiently generates high-quality CGRA architectures, significantly reducing manual design effort and demonstrating the potential of our framework for real-world CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21700v3">XBreaking: Understanding how LLMs security alignment can be broken</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. These mechanisms maintain the integrity of LLM alignment by guaranteeing that the models respond safely and ethically. In response to this, attacks on LLMs are a significant threat to such protections, and many previous approaches have already demonstrated their effectiveness across diverse domains. Existing LLM attacks mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel approach that exploits these unique patterns to break the security and alignment constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05311v1">Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05269v1">TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted at ICML 2025 MAS Workshop. This version includes additional experiments and analysis
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14450v2">LLM4FaaS: No-Code Application Development using LLMs and FaaS</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted for publication in 2025 IEEE/ACM 18th International Conference on Utility and Cloud Computing
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show great capabilities in generating code from natural language descriptions, bringing programming power closer to non-technical users. However, their lack of expertise in operating the generated code remains a key barrier to realizing customized applications. Function-as-a-Service (FaaS) platforms offer a high level of abstraction for code execution and deployment, allowing users to run LLM-generated code without requiring technical expertise or incurring operational overhead. In this paper, we present LLM4FaaS, a no-code application development approach that integrates LLMs and FaaS platforms to enable non-technical users to build and run customized applications using only natural language. By deploying LLM-generated code through FaaS, LLM4FaaS abstracts away infrastructure management and boilerplate code generation. We implement a proof-of-concept prototype based on an open-source FaaS platform, and evaluate it using real prompts from non-technical users. Experiments with GPT-4o show that LLM4FaaS can automatically build and deploy code in 71.47% of cases, outperforming a non-FaaS baseline at 43.48% and an existing LLM-based platform at 14.55%, narrowing the gap to human performance at 88.99%. Further analysis of code quality, programming language diversity, latency, and consistency demonstrates a balanced performance in terms of efficiency, maintainability and availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17086v4">Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 EMNLP 2025 Oral
    </div>
    <details class="paper-abstract">
      Peer review underpins scientific progress, but it is increasingly strained by reviewer shortages and growing workloads. Large Language Models (LLMs) can automatically draft reviews now, but determining whether LLM-generated reviews are trustworthy requires systematic evaluation. Researchers have evaluated LLM reviews at either surface-level (e.g., BLEU and ROUGE) or content-level (e.g., specificity and factual accuracy). Yet it remains uncertain whether LLM-generated reviews attend to the same critical facets that human experts weigh -- the strengths and weaknesses that ultimately drive an accept-or-reject decision. We introduce a focus-level evaluation framework that operationalizes the focus as a normalized distribution of attention across predefined facets in paper reviews. Based on the framework, we developed an automatic focus-level evaluation pipeline based on two sets of facets: target (e.g., problem, method, and experiment) and aspect (e.g., validity, clarity, and novelty), leveraging 676 paper reviews (https://figshare.com/s/d5adf26c802527dd0f62) from OpenReview that consists of 3,657 strengths and weaknesses identified from human experts. The comparison of focus distributions between LLMs and human experts showed that the off-the-shelf LLMs consistently have a more biased focus towards examining technical validity while significantly overlooking novelty assessment when criticizing papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13142v3">Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Codebase: https://github.com/EvolvingLMMs-Lab/EASI/
    </div>
    <details class="paper-abstract">
      Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence. We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a standardized protocol for the fair evaluation of state-of-the-art proprietary and open-source models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence (SI), yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail even the most advanced multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18469v6">Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to NAACL 2025 Main (Oral)
    </div>
    <details class="paper-abstract">
      Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25441v2">Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 27 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel as passive responders, but teaching them to be proactive, goal-oriented partners, a critical capability in high-stakes domains, remains a major challenge. Current paradigms either myopically optimize single-turn attributes or rely on brittle, high-cost user simulators, creating a persistent ``reality gap''. To bridge this gap, we introduce \texttt{Learn-to-Ask}, a general, simulator-free framework for learning and deploying proactive dialogue agents \textit{directly from offline expert data}, bypassing the need to model complex user dynamics. Our key insight is to reframe the offline policy learning problem by leveraging the \textbf{observed future} of each expert trajectory. This allows us to infer a dense, turn-by-turn reward signal grounded in the expert's revealed strategy, decomposing the intractable long-horizon problem into a series of supervised learning tasks, and training a policy to output a structured \texttt{(action, state_assessment)} tuple, governing both \textbf{what to ask} and, crucially, \textbf{when to stop}. To ensure reward fidelity, our Automated Grader Calibration pipeline systematically purges noise from the LLM-based reward model with minimal human supervision. Empirically, we demonstrate the efficacy of \texttt{Learn-to-Ask} in a real-world medical dataset, using LLMs of varying sizes up to 32B. Our approach culminates in the successful deployment of LLMs into a live, large-scale online AI service. In rigorous in-house evaluations, our model was launched and achieved performance even superior to human experts, proving our framework's ability to translate offline data into tangible, real-world impact. We hope this work provides a practical and economically viable blueprint for transforming passive LLMs into proactive, goal-oriented LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16447v2">Boardwalk: Towards a Framework for Creating Board Games with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Presented at SBGames 2025
    </div>
    <details class="paper-abstract">
      Implementing board games in code can be a time-consuming task. However, Large Language Models (LLMs) have been proven effective at generating code for domain-specific tasks with simple contextual information. We aim to investigate whether LLMs can implement digital versions of board games from rules described in natural language. This would be a step towards an LLM-assisted framework for quick board game code generation. We expect to determine the main challenges for LLMs to implement the board games, and how different approaches and models compare to one another. We task three state-of-the-art LLMs (Claude, DeepSeek and ChatGPT) with coding a selection of 12 popular and obscure games in free-form and within Boardwalk, our proposed General Game Playing API. We anonymize the games and components to avoid evoking pre-trained LLM knowledge. The implementations are tested for playability and rule compliance. We evaluate success rate and common errors across LLMs and game popularity. Our approach proves viable, with the best performing model, Claude 3.7 Sonnet, yielding 55.6\% of games without any errors. While compliance with the API increases error frequency, the severity of errors is more significantly dependent on the LLM. We outline future steps for creating a framework to integrate this process, making the elaboration of board games more accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05114v1">Usando LLMs para Programar Jogos de Tabuleiro e Variações</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted for presentation at the I Escola Regional de Aprendizado de M\'aquina e Intelig\^encia Artificial da Regi\~ao Sul, 2025, in Portuguese language
    </div>
    <details class="paper-abstract">
      Creating programs to represent board games can be a time-consuming task. Large Language Models (LLMs) arise as appealing tools to expedite this process, given their capacity to efficiently generate code from simple contextual information. In this work, we propose a method to test how capable three LLMs (Claude, DeepSeek and ChatGPT) are at creating code for board games, as well as new variants of existing games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22963v2">CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05080v1">On Text Simplification Metrics and General-Purpose LLMs for Accessible Health Information, and A Potential Architectural Advantage of The Instruction-Tuned LLM class</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      The increasing health-seeking behavior and digital consumption of biomedical information by the general public necessitate scalable solutions for automatically adapting complex scientific and technical documents into plain language. Automatic text simplification solutions, including advanced large language models, however, continue to face challenges in reliably arbitrating the tension between optimizing readability performance and ensuring preservation of discourse fidelity. This report empirically assesses the performance of two major classes of general-purpose LLMs, demonstrating their linguistic capabilities and foundational readiness for the task compared to a human benchmark. Using a comparative analysis of the instruction-tuned Mistral 24B and the reasoning-augmented QWen2.5 32B, we identify a potential architectural advantage in the instruction-tuned LLM. Mistral exhibits a tempered lexical simplification strategy that enhances readability across a suite of metrics and the simplification-specific formula SARI (mean 42.46), while preserving human-level discourse with a BERTScore of 0.91. QWen also attains enhanced readability performance, but its operational strategy shows a disconnect in balancing between readability and accuracy, reaching a statistically significantly lower BERTScore of 0.89. Additionally, a comprehensive correlation analysis of 21 metrics spanning readability, discourse fidelity, content safety, and underlying distributional measures for mechanistic insights, confirms strong functional redundancies among five readability indices. This empirical evidence tracks baseline performance of the evolving LLMs for the task of text simplification, identifies the instruction-tuned Mistral 24B for simplification, provides necessary heuristics for metric selection, and points to lexical support as a primary domain-adaptation issue for simplification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21150v2">String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      We introduce String Seed of Thought (SSoT), a novel prompting method for LLMs that improves Probabilistic Instruction Following (PIF). We define PIF as a task requiring an LLM to select its answer from a predefined set of options, each associated with a specific probability, such that the empirical distribution of the generated answers aligns with the target distribution when prompted multiple times. While LLMs excel at tasks with single, deterministic answers, they often fail at PIF, exhibiting biases problematic for applications requiring non-deterministic behaviors, such as human-behavior simulation, content diversification, and multiplayer games. It also harms the diversity of generated responses, a crucial factor in test-time scaling, by causing the outputs to collapse into a limited set of answers. To address this, we propose SSoT, a simple prompting method that instructs an LLM to first output a random string to generate sufficient entropy. SSoT also instructs the LLM to extract randomness by manipulating this string to derive a final answer, thereby preserving diversity while adhering to specific constraints. We demonstrate that SSoT significantly improves the PIF performance of LLMs, approaching the ideal performance of a pseudo-random number generator. Furthermore, our experiments on NoveltyBench show SSoT's benefits extend beyond closed-set tasks to open-ended tasks by enhancing response diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v5">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted as a spotlight at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling. Code is available at https://github.com/SakanaAI/treequest .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16395v3">AIRepr: An Analyst-Inspector Framework for Evaluating Reproducibility of LLMs in Data Science</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to 2025 EMNLP findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate data analysis through executable code generation. Yet, data science tasks often admit multiple statistically valid solutions, e.g. different modeling strategies, making it critical to understand the reasoning behind analyses, not just their outcomes. While manual review of LLM-generated code can help ensure statistical soundness, it is labor-intensive and requires expertise. A more scalable approach is to evaluate the underlying workflows-the logical plans guiding code generation. However, it remains unclear how to assess whether an LLM-generated workflow supports reproducible implementations. To address this, we present AIRepr, an Analyst-Inspector framework for automatically evaluating and improving the reproducibility of LLM-generated data analysis workflows. Our framework is grounded in statistical principles and supports scalable, automated assessment. We introduce two novel reproducibility-enhancing prompting strategies and benchmark them against standard prompting across 15 analyst-inspector LLM pairs and 1,032 tasks from three public benchmarks. Our findings show that workflows with higher reproducibility also yield more accurate analyses, and that reproducibility-enhancing prompts substantially improve both metrics. This work provides a foundation for transparent, reliable, and efficient human-AI collaboration in data science. Our code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04962v1">Too Good to be Bad: On the Failure of LLMs to Role-Play Villains</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01602v3">L2T-Tune:LLM-Guided Hybrid Database Tuning with LHS and TD3</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Configuration tuning is critical for database performance. Although recent advancements in database tuning have shown promising results in throughput and latency improvement, challenges remain. First, the vast knob space makes direct optimization unstable and slow to converge. Second, reinforcement learning pipelines often lack effective warm-start guidance and require long offline training. Third, transferability is limited: when hardware or workloads change, existing models typically require substantial retraining to recover performance. To address these limitations, we propose L2T-Tune, a new LLM-guided hybrid database tuning framework that features a three-stage pipeline: Stage one performs a warm start that simultaneously generates uniform samples across the knob space and logs them into a shared pool; Stage two leverages a large language model to mine and prioritize tuning hints from manuals and community documents for rapid convergence. Stage three uses the warm-start sample pool to reduce the dimensionality of knobs and state features, then fine-tunes the configuration with the Twin Delayed Deep Deterministic Policy Gradient algorithm. We conduct experiments on L2T-Tune and the state-of-the-art models. Compared with the best-performing alternative, our approach improves performance by an average of 37.1% across all workloads, and by up to 73% on TPC-C. Compared with models trained with reinforcement learning, it achieves rapid convergence in the offline tuning stage on a single server. Moreover, during the online tuning stage, it only takes 30 steps to achieve best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04934v1">Leak@$k$: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) is critical for regulatory compliance and for building ethical generative AI systems that avoid producing private, toxic, illegal, or copyrighted content. Despite rapid progress, in this work we show that \textit{almost all} existing unlearning methods fail to achieve true forgetting in practice. Specifically, while evaluations of these `unlearned' models under deterministic (greedy) decoding often suggest successful knowledge removal using standard benchmarks (as has been done in the literature), we show that sensitive information reliably resurfaces when models are sampled with standard probabilistic decoding. To rigorously capture this vulnerability, we introduce \texttt{leak@$k$}, a new meta-evaluation metric that quantifies the likelihood of forgotten knowledge reappearing when generating $k$ samples from the model under realistic decoding strategies. Using three widely adopted benchmarks, TOFU, MUSE, and WMDP, we conduct the first large-scale, systematic study of unlearning reliability using our newly defined \texttt{leak@$k$} metric. Our findings demonstrate that knowledge leakage persists across methods and tasks, underscoring that current state-of-the-art unlearning techniques provide only limited forgetting and highlighting the urgent need for more robust approaches to LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20514v2">SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Topic discovery in scientific literature provides valuable insights for researchers to identify emerging trends and explore new avenues for investigation, facilitating easier scientific information retrieval. Many machine learning methods, particularly deep embedding techniques, have been applied to discover research topics. However, most existing topic discovery methods rely on word embedding to capture the semantics and lack a comprehensive understanding of scientific publications, struggling with complex, high-dimensional text relationships. Inspired by the exceptional comprehension of textual information by large language models (LLMs), we propose an advanced topic discovery method enhanced by LLMs to improve scientific topic identification, namely SciTopic. Specifically, we first build a textual encoder to capture the content from scientific publications, including metadata, title, and abstract. Next, we construct a space optimization module that integrates entropy-based sampling and triplet tasks guided by LLMs, enhancing the focus on thematic relevance and contextual intricacies between ambiguous instances. Then, we propose to fine-tune the textual encoder based on the guidance from the LLMs by optimizing the contrastive loss of the triplets, forcing the text encoder to better discriminate instances of different topics. Finally, extensive experiments conducted on three real-world datasets of scientific publications demonstrate that SciTopic outperforms the state-of-the-art (SOTA) scientific topic discovery methods, enabling researchers to gain deeper and faster insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04921v1">AgentExpt: Automating AI Experiment Design with LLM-based Resource Retrieval Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language model agents are becoming increasingly capable at web-centric tasks such as information retrieval, complex reasoning. These emerging capabilities have given rise to surge research interests in developing LLM agent for facilitating scientific quest. One key application in AI research is to automate experiment design through agentic dataset and baseline retrieval. However, prior efforts suffer from limited data coverage, as recommendation datasets primarily harvest candidates from public portals and omit many datasets actually used in published papers, and from an overreliance on content similarity that biases model toward superficial similarity and overlooks experimental suitability. Harnessing collective perception embedded in the baseline and dataset citation network, we present a comprehensive framework for baseline and dataset recommendation. First, we design an automated data-collection pipeline that links roughly one hundred thousand accepted papers to the baselines and datasets they actually used. Second, we propose a collective perception enhanced retriever. To represent the position of each dataset or baseline within the scholarly network, it concatenates self-descriptions with aggregated citation contexts. To achieve efficient candidate recall, we finetune an embedding model on these representations. Finally, we develop a reasoning-augmented reranker that exact interaction chains to construct explicit reasoning chains and finetunes a large language model to produce interpretable justifications and refined rankings. The dataset we curated covers 85\% of the datasets and baselines used at top AI conferences over the past five years. On our dataset, the proposed method outperforms the strongest prior baseline with average gains of +5.85\% in Recall@20, +8.30\% in HitRate@5. Taken together, our results advance reliable, interpretable automation of experimental design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04157v1">Are We Aligned? A Preliminary Investigation of the Alignment of Responsible AI Values between LLMs and Human Judgment</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in software engineering tasks such as requirements elicitation, design, and evaluation, raising critical questions regarding their alignment with human judgments on responsible AI values. This study investigates how closely LLMs' value preferences align with those of two human groups: a US-representative sample and AI practitioners. We evaluate 23 LLMs across four tasks: (T1) selecting key responsible AI values, (T2) rating their importance in specific contexts, (T3) resolving trade-offs between competing values, and (T4) prioritizing software requirements that embody those values. The results show that LLMs generally align more closely with AI practitioners than with the US-representative sample, emphasizing fairness, privacy, transparency, safety, and accountability. However, inconsistencies appear between the values that LLMs claim to uphold (Tasks 1-3) and the way they prioritize requirements (Task 4), revealing gaps in faithfulness between stated and applied behavior. These findings highlight the practical risk of relying on LLMs in requirements engineering without human oversight and motivate the need for systematic approaches to benchmark, interpret, and monitor value alignment in AI-assisted software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04136v1">Implementation of transformer-based LLMs with large-scale optoelectronic neurons on a CMOS image sensor platform</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      The recent rapid deployment of datacenter infrastructures for performing large language models (LLMs) and related artificial intelligence (AI) applications in the clouds is predicted to incur an exponentially growing energy consumption in the near-term future. In this paper, we propose and analyze the implementation of the transformer model, which is the cornerstone of the modern LLMs, with novel large-scale optoelectronic neurons (OENs) constructed over the commercially available complementary metal-oxide-semiconductor (CMOS) image sensor (CIS) platform. With all of the required optoelectronic devices and electronic circuits integrated in a chiplet only about 2 cm by 3 cm in size, 175 billon parameters in the case of GPT-3 are shown to perform inference at an unprecedented speed of 12.6 POPS using only a 40 nm CMOS process node, along with a high power efficiency of 74 TOPS/W and a high area efficiency of 19 TOPS/mm2, both surpassing the related digital electronics by roughly two orders of magnitude. The influence of the quantization formats and the hardware induced errors are numerically investigated, and are shown to have a minimal impact. Our study presents a new yet practical path toward analog neural processing units (NPUs) to complement existing digital processing units.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04087v1">E-CARE: An Efficient LLM-based Commonsense-Augmented Framework for E-Commerce</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Finding relevant products given a user query plays a pivotal role in an e-commerce platform, as it can spark shopping behaviors and result in revenue gains. The challenge lies in accurately predicting the correlation between queries and products. Recently, mining the cross-features between queries and products based on the commonsense reasoning capacity of Large Language Models (LLMs) has shown promising performance. However, such methods suffer from high costs due to intensive real-time LLM inference during serving, as well as human annotations and potential Supervised Fine Tuning (SFT). To boost efficiency while leveraging the commonsense reasoning capacity of LLMs for various e-commerce tasks, we propose the Efficient Commonsense-Augmented Recommendation Enhancer (E-CARE). During inference, models augmented with E-CARE can access commonsense reasoning with only a single LLM forward pass per query by utilizing a commonsense reasoning factor graph that encodes most of the reasoning schema from powerful LLMs. The experiments on 2 downstream tasks show an improvement of up to 12.1% on precision@5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04064v1">Benchmarking and Studying the LLM-based Agent System in End-to-End Software Development</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      The development of LLM-based autonomous agents for end-to-end software development represents a significant paradigm shift in software engineering. However, the scientific evaluation of these systems is hampered by significant challenges, including overly simplistic benchmarks and the difficulty of conducting fair comparisons between different agent architectures due to confounding implementation variables. To address these limitations, we first construct a challenging and dynamically curated E2EDevBench to simulate realistic development scenarios. Second, we propose a hybrid evaluation framework that combines test-case-based functional assessment with fine-grained, LLM-based requirement verification. Using this framework, we conduct a controlled empirical study on three representative agent architectures implemented upon a unified foundation to isolate the impact of workflow design. Our findings reveal that state-of-the-art agents can fulfill approximately 50\% of requirements on \bench{}, but their success is critically dependent on the architectural strategy for task decomposition and collaboration. Furthermore, our analysis indicates that the primary bottleneck is the omission of requirements and inadequate self-verification. This work provides the community with a more realistic benchmark, a comprehensive evaluation framework, and crucial insights into the current capabilities and core challenges of software development agents, guiding future research toward enhancing requirement comprehension and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00730v2">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 This work is intended for submission to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04042v1">An LLM-based Framework for Human-Swarm Teaming Cognition in Disaster Search and Rescue</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large-scale disaster Search And Rescue (SAR) operations are persistently challenged by complex terrain and disrupted communications. While Unmanned Aerial Vehicle (UAV) swarms offer a promising solution for tasks like wide-area search and supply delivery, yet their effective coordination places a significant cognitive burden on human operators. The core human-machine collaboration bottleneck lies in the ``intention-to-action gap'', which is an error-prone process of translating a high-level rescue objective into a low-level swarm command under high intensity and pressure. To bridge this gap, this study proposes a novel LLM-CRF system that leverages Large Language Models (LLMs) to model and augment human-swarm teaming cognition. The proposed framework initially captures the operator's intention through natural and multi-modal interactions with the device via voice or graphical annotations. It then employs the LLM as a cognitive engine to perform intention comprehension, hierarchical task decomposition, and mission planning for the UAV swarm. This closed-loop framework enables the swarm to act as a proactive partner, providing active feedback in real-time while reducing the need for manual monitoring and control, which considerably advances the efficacy of the SAR task. We evaluate the proposed framework in a simulated SAR scenario. Experimental results demonstrate that, compared to traditional order and command-based interfaces, the proposed LLM-driven approach reduced task completion time by approximately $64.2\%$ and improved task success rate by $7\%$. It also leads to a considerable reduction in subjective cognitive workload, with NASA-TLX scores dropping by $42.9\%$. This work establishes the potential of LLMs to create more intuitive and effective human-swarm collaborations in high-stakes scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04036v1">PICNIC: Silicon Photonic Interconnected Chiplets with Computational Network and In-memory Computing for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      This paper presents a 3D-stacked chiplets based large language model (LLM) inference accelerator, consisting of non-volatile in-memory-computing processing elements (PEs) and Inter-PE Computational Network (IPCN), interconnected via silicon photonic to effectively address the communication bottlenecks. A LLM mapping scheme was developed to optimize hardware scheduling and workload mapping. Simulation results show it achieves $3.95\times$ speedup and $30\times$ efficiency improvement over the Nvidia A100 before chiplet clustering and power gating scheme (CCPG). Additionally, the system achieves further scalability and efficiency improvement with the implementation of CCPG to accommodate larger models, attaining $57\times$ efficiency improvement over Nvidia H100 at similar throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27140v2">Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04023v1">LLM-Driven Adaptive Source-Sink Identification and False Positive Mitigation for Static Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Static analysis is effective for discovering software vulnerabilities but notoriously suffers from incomplete source--sink specifications and excessive false positives (FPs). We present \textsc{AdaTaint}, an LLM-driven taint analysis framework that adaptively infers source/sink specifications and filters spurious alerts through neuro-symbolic reasoning. Unlike LLM-only detectors, \textsc{AdaTaint} grounds model suggestions in program facts and constraint validation, ensuring both adaptability and determinism. We evaluate \textsc{AdaTaint} on Juliet 1.3, SV-COMP-style C benchmarks, and three large real-world projects. Results show that \textsc{AdaTaint} reduces false positives by \textbf{43.7\%} on average and improves recall by \textbf{11.2\%} compared to state-of-the-art baselines (CodeQL, Joern, and LLM-only pipelines), while maintaining competitive runtime overhead. These findings demonstrate that combining LLM inference with symbolic validation offers a practical path toward more accurate and reliable static vulnerability analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02263v3">LA-MARRVEL: A Knowledge-Grounded and Language-Aware LLM Reranker for AI-MARRVEL in Rare Disease Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Diagnosing rare diseases requires linking gene findings with often unstructured reference text. Current pipelines collect many candidate genes, but clinicians still spend a lot of time filtering false positives and combining evidence from papers and databases. A key challenge is language: phenotype descriptions and inheritance patterns are written in prose, not fully captured by tables. Large language models (LLMs) can read such text, but clinical use needs grounding in citable knowledge and stable, repeatable behavior. We explore a knowledge-grounded and language-aware reranking layer on top of a high-recall first-stage pipeline. The goal is to improve precision and explainability, not to replace standard bioinformatics steps. We use expert-built context and a consensus method to reduce LLM variability, producing shorter, better-justified gene lists for expert review. LA-MARRVEL achieves the highest accuracy, outperforming other methods -- including traditional bioinformatics diagnostic tools (AI-MARRVEL, Exomiser, LIRICAL) and naive large language models (e.g., Anthropic Claude) -- with an average Recall@5 of 94.10%, a +3.65 percentage-point improvement over AI-MARRVEL. The LLM-generated reasoning provides clear prose on phenotype matching and inheritance patterns, making clinical review faster and easier. LA-MARRVEL has three parts: expert-engineered context that enriches phenotype and disease information; a ranked voting algorithm that combines multiple LLM runs to choose a consensus ranked gene list; and the AI-MARRVEL pipeline that provides first-stage ranks and gene annotations, already known as a state-of-the-art method in Rare Disease Diagnosis on BG, DDD, and UDN cohorts. The online AI-MARRVEL includes LA-MARRVEL as an LLM feature at https://ai.marrvel.org . We evaluate LA-MARRVEL on three datasets from independent cohorts of real-world diagnosed patients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00847v3">Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $\epsilon\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-\epsilon}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings.
    </details>
</div>
