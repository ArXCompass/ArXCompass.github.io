# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08872v3">GTAlign: Game-Theoretic Alignment of LLM Assistants for Social Welfare</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a social welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and social welfare compared to baselines across diverse tasks. The code is available at https://github.com/ulab-uiuc/GTAlign .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v6">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20172v3">Benchmarking LLMs in Web API Integration Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 To be published in Proceedings of 2025 2nd IEEE/ACM International Conference on AI-powered Software (AIware), Data & Benchmark Track; switched to IEEE conference template
    </div>
    <details class="paper-abstract">
      API integration is a cornerstone of our digital infrastructure, enabling software systems to connect and interact. However, as shown by many studies, writing or generating correct code to invoke APIs, particularly web APIs, is challenging. Although large language models (LLMs) have become popular in software development, their effectiveness in automating the generation of web API integration code remains unexplored. In order to address this, we present WAPIIBench, a dataset and evaluation pipeline designed to assess the ability of LLMs to generate web API invocation code. Our experiments with several open-source LLMs reveal that generating API invocations poses a significant challenge, resulting in hallucinated endpoints, incorrect argument usage, and other errors. None of the evaluated open-source models was able to solve more than 40% of the tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06857v6">What is the Role of Small Models in the LLM Era: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 a survey paper of small models
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in advancing artificial general intelligence (AGI), leading to the development of increasingly large models such as GPT-4 and LLaMA-405B. However, scaling up model sizes results in exponentially higher computational costs and energy consumption, making these models impractical for academic researchers and businesses with limited resources. At the same time, Small Models (SMs) are frequently used in practical settings, although their significance is currently underestimated. This raises important questions about the role of small models in the era of LLMs, a topic that has received limited attention in prior research. In this work, we systematically examine the relationship between LLMs and SMs from two key perspectives: Collaboration and Competition. We hope this survey provides valuable insights for practitioners, fostering a deeper understanding of the contribution of small models and promoting more efficient use of computational resources. The code is available at https://github.com/tigerchen52/role_of_small_models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21590v2">Representation Consistency for Accurate and Coherent LLM Answer Aggregation</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 Accepted at NeurIPS 2025. Camera-ready version
    </div>
    <details class="paper-abstract">
      Test-time scaling improves large language models' (LLMs) performance by allocating more compute budget during inference. To achieve this, existing methods often require intricate modifications to prompting and sampling strategies. In this work, we introduce representation consistency (RC), a test-time scaling method for aggregating answers drawn from multiple candidate responses of an LLM regardless of how they were generated, including variations in prompt phrasing and sampling strategy. RC enhances answer aggregation by not only considering the number of occurrences of each answer in the candidate response set, but also the consistency of the model's internal activations while generating the set of responses leading to each answer. These activations can be either dense (raw model activations) or sparse (encoded via pretrained sparse autoencoders). Our rationale is that if the model's representations of multiple responses converging on the same answer are highly variable, this answer is more likely to be the result of incoherent reasoning and should be down-weighted during aggregation. Importantly, our method only uses cached activations and lightweight similarity computations and requires no additional model queries. Through experiments with four open-source LLMs and four reasoning datasets, we validate the effectiveness of RC for improving task performance during inference, with consistent accuracy improvements (up to 4%) over strong test-time scaling baselines. We also show that consistency in the sparse activation signals aligns well with the common notion of coherent reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27598v2">InnovatorBench: Evaluating Agents' Ability to Conduct Innovative LLM Research</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      AI agents could accelerate scientific discovery by automating hypothesis formation, experiment design, coding, execution, and analysis, yet existing benchmarks probe narrow skills in simplified settings. To address this gap, we introduce InnovatorBench, a benchmark-platform pair for realistic, end-to-end assessment of agents performing Large Language Model (LLM) research. It comprises 20 tasks spanning Data Construction, Filtering, Augmentation, Loss Design, Reward Design, and Scaffold Construction, which require runnable artifacts and assessment of correctness, performance, output quality, and uncertainty. To support agent operation, we develop ResearchGym, a research environment offering rich action spaces, distributed and long-horizon execution, asynchronous monitoring, and snapshot saving. We also implement a lightweight ReAct agent that couples explicit reasoning with executable planning using frontier models such as Claude-4, GPT-5, GLM-4.5, and Kimi-K2. Our experiments demonstrate that while frontier models show promise in code-driven research tasks, they struggle with fragile algorithm-related tasks and long-horizon decision making, such as impatience, poor resource management, and overreliance on template-based reasoning. Furthermore, agents require over 11 hours to achieve their best performance on InnovatorBench, underscoring the benchmark's difficulty and showing the potential of InnovatorBench to be the next generation of code-based research benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14846v3">Where to Search: Measure the Prior-Structured Search Space of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 11 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via two instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03824v2">Memory Assisted LLM for Personalized Recommendation System</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in solving recommendation tasks. With proven capabilities in understanding user preferences, LLM personalization has emerged as a critical area for providing tailored responses to individuals. Current studies explore personalization through prompt design and fine-tuning, paving the way for further research in personalized LLMs. However, existing approaches are either costly and inefficient in capturing diverse user preferences or fail to account for timely updates to user history. To address these gaps, we propose the Memory-Assisted Personalized LLM (MAP). Through user interactions, we first create a history profile for each user, capturing their preferences, such as ratings for historical items. During recommendation, we extract relevant memory based on similarity, which is then incorporated into the prompts to enhance personalized recommendations. In our experiments, we define a new task that enables testing with varying memory size under two scenarios: single domain where memory and tasks are from the same category and cross-domain (e.g. memory from movies and recommendation tasks in books). The results show that MAP outperforms regular LLM-based recommenders that integrate user history directly through prompt design. Moreover, as user history grows, MAP's advantage increases in both scenarios, making it more suitable for addressing successive personalized user requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13790v2">Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13500v2">MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      LLMs hold great promise for healthcare applications, but the rapid evolution of medical knowledge and errors in training data often cause them to generate outdated or inaccurate information, limiting their applicability in high-stakes clinical practice. Model editing has emerged as a potential remedy without full retraining. While parameter-based editing often compromises locality and is thus ill-suited for the medical domain, retrieval-based editing offers a more viable alternative. However, it still faces two critical challenges: (1) representation overlap within the medical knowledge space often causes inaccurate retrieval and reduces editing accuracy; (2) existing methods are restricted to single-sample edits, while batch-editing remains largely unexplored despite its importance for real-world medical applications. To address these challenges, we first construct MedVersa, an enhanced benchmark with broader coverage of medical subjects, designed to evaluate both single and batch edits under strict locality constraints. We then propose MedREK, a retrieval-based editing framework that integrates a shared query-key module for precise matching with an attention-based prompt encoder for informative guidance. Experimental results on various medical benchmarks demonstrate that our MedREK achieves superior performance across different core metrics and provides the first validated solution for batch-editing in medical LLMs. Our code and dataset are available at https://github.com/mylittleriver/MedREK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09802v2">Enhancing Reasoning Abilities of Small LLMs with Cognitive Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 emnlp 2025 main conference
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need for training effective small reasoning models. A critical challenge is that small models possess different reasoning capacities and cognitive trajectories compared with their larger counterparts. Hence, directly distilling chain-of-thought (CoT) rationales from large LRMs to smaller ones can sometimes be ineffective and often requires a substantial amount of annotated data. In this paper, we first introduce a novel Critique-Rethink-Verify (CRV) system, designed for training smaller yet powerful LRMs. Our CRV system consists of multiple LLM agents, each specializing in unique tasks: (i) critiquing the CoT rationales according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. Building on the CRV system, we further propose the Cognitive Preference Optimization (CogPO) algorithm to continuously enhance the reasoning abilities of smaller models by aligning their reasoning processes with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of our CRV+CogPO framework, which outperforms other methods by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04405v2">FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate exceptional performance but entail significant memory and computational costs, restricting their practical deployment. While existing INT4/INT8 quantization reduces these costs, they often degrade accuracy or lack optimal efficiency. INT6 quantization offers a superior trade-off between model accuracy and inference efficiency, but lacks hardware support in modern GPUs, forcing emulation via higher-precision arithmetic units that limit acceleration. In this paper, we propose FlexQ, a novel post-training INT6 quantization framework combining algorithmic innovation with system-level optimizations. FlexQ employs uniform 6-bit weight quantization across all layers, with adaptive retention of 8-bit activations in layers identified through layer-wise sensitivity analysis. To maximize hardware efficiency, we develop a specialized high-performance GPU kernel supporting matrix multiplication for W6A6 and W6A8 representations via Binary Tensor Core (BTC) equivalents, effectively bypassing the lack of native INT6 tensor cores. Evaluations on LLaMA family models show FlexQ maintains near-FP16 accuracy, with perplexity increases of no more than 0.1 on WikiText2. The proposed kernel achieves an average 1.39$\times$ speedup over ABQ-LLM on LLaMA-2-70B linear layers. End-to-end, FlexQ delivers 1.33$\times$ inference acceleration and 1.21$\times$ memory savings over SmoothQuant. Code is released at https://github.com/FlyFoxPlayer/FlexQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11671v3">Computational Basis of LLM's Decision Making in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract "vectors of variable variations" (e.g., "male" to "female") from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications, strengthening sociological theory and measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10987v2">DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 14 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at https://github.com/hsannn/ditto.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13918v3">FTSmartAudit: A Knowledge Distillation-Enhanced Framework for Automated Smart Contract Auditing Using Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of blockchain technology has driven the widespread adoption of smart contracts. However, their inherent vulnerabilities have led to significant financial losses. Traditional auditing methods, while essential, struggle to keep pace with the increasing complexity and scale of smart contracts. Large Language Models (LLMs) offer promising capabilities for automating vulnerability detection, but their adoption is often limited by high computational costs. Although prior work has explored leveraging large models through agents or workflows, relatively little attention has been given to improving the performance of smaller, fine-tuned models--a critical factor for achieving both efficiency and data privacy. In this paper, we introduce HKT-SmartAudit, a framework for developing lightweight models optimized for smart contract auditing. It features a multi-stage knowledge distillation pipeline that integrates classical distillation, external domain knowledge, and reward-guided learning to transfer high-quality insights from large teacher models. A single-task learning strategy is employed to train compact student models that maintain high accuracy and robustness while significantly reducing computational overhead. Experimental results show that our distilled models outperform both commercial tools and larger models in detecting complex vulnerabilities and logical flaws, offering a practical, secure, and scalable solution for smart contract auditing. The source code is available at Github repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12815v2">Learning to Steer: Input-dependent Steering for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Steering has emerged as a practical approach to enable post-hoc guidance of LLMs towards enforcing a specific behavior. However, it remains largely underexplored for multimodal LLMs (MLLMs); furthermore, existing steering techniques, such as mean steering, rely on a single steering vector, applied independently of the input query. This paradigm faces limitations when the desired behavior is dependent on the example at hand. For example, a safe answer may consist in abstaining from answering when asked for an illegal activity, or may point to external resources or consultation with an expert when asked about medical advice. In this paper, we investigate a fine-grained steering that uses an input-specific linear shift. This shift is computed using contrastive input-specific prompting. However, the input-specific prompts required for this approach are not known at test time. Therefore, we propose to train a small auxiliary module to predict the input-specific steering vector. Our approach, dubbed as L2S (Learn-to-Steer), demonstrates that it reduces hallucinations and enforces safety in MLLMs, outperforming other static baselines. Our code is publicly available at https://jayneelparekh.github.io/learn-to-steer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04103v3">How to Train Your LLM Web Agent: A Statistical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v4">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10142v3">Debiasing LLMs by Masking Unfairness-Driving Attention Heads</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly mediate decisions in domains where unfair treatment of demographic groups is unacceptable. Existing work probes when biased outputs appear, but gives little insight into the mechanisms that generate them, leaving existing mitigations largely fragile. In this paper, we conduct a systematic investigation LLM unfairness and propose DiffHeads, a lightweight debiasing framework for LLMs. We first compare Direct-Answer (DA) prompting to Chain-of-Thought (CoT) prompting across eight representative open- and closed-source LLMs. DA will trigger the nature bias part of LLM and improve measured unfairness by 534.5%-391.9% in both one-turn and two-turn dialogues. Next, we define a token-to-head contribution score that traces each token's influence back to individual attention heads. This reveals a small cluster of bias heads that activate under DA but stay largely dormant with CoT, providing the first causal link between prompting strategy and bias emergence. Finally, building on this insight, we propose DiffHeads that identifies bias heads through differential activation analysis between DA and CoT, and selectively masks only those heads. DiffHeads reduces unfairness by 49.4%, and 40.3% under DA and CoT, respectively, without harming model utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18253v2">From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature. To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance. Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22603v2">Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05286v2">HEXGEN-FLOW: Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Recent advancements in leveraging the agentic paradigm of large language models (LLMs) have substantially improved Text-to-SQL capabilities, empowering users without specialized database knowledge to intuitively query databases. However, deploying agentic LLM-based Text-to-SQL systems in production presents significant challenges, stemming from their inherently multi-stage computational dependencies, strict latency requirements, and the complexity of deployment across heterogeneous GPUs widely existing in enterprise clusters. Meanwhile, existing LLM serving frameworks are primarily designed for independent inference tasks, resulting in suboptimal performance and frequent service-level objective (SLO) violations in Text-to-SQL workloads. In this paper, we introduce HEXGEN-FLOW, a novel framework designed explicitly to schedule and execute agentic multi-stage LLM-based Text-to-SQL workflows on heterogeneous GPU clusters serving multi-tenant Text-to-SQL requests. HEXGEN-FLOW introduces a hierarchical scheduling approach that combines global workload-balanced task dispatching with an adaptive local priority queue, guided by a systematic analysis of agentic Text-to-SQL workflows. Additionally, we propose a lightweight simulation-based method for tuning critical scheduling hyperparameters, further enhancing robustness and adaptability. Our evaluation on realistic Text-to-SQL benchmarks demonstrates that HEXGEN-FLOW significantly outperforms state-of-the-art LLM serving frameworks. Across all traces, HEXGEN-FLOW reduces P95 tail latency by $1.42{\sim}1.56\times$ and increases throughput by $1.49{\sim}1.81\times$, demonstrating robust improvements under diverse workloads. Our code is available at https://github.com/Relaxed-System-Lab/Hexgen-Flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19354v2">The Emerged Security and Privacy of LLM Agent: A Survey with Case Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 35 pages, 19 figures. Accepted to ACM Computing Surveys (CSUR), 2025
    </div>
    <details class="paper-abstract">
      Inspired by the rapid development of Large Language Models (LLMs), LLM agents have evolved to perform complex tasks. LLM agents are now extensively applied across various domains, handling vast amounts of data to interact with humans and execute tasks. The widespread applications of LLM agents demonstrate their significant commercial value; however, they also expose security and privacy vulnerabilities. At the current stage, comprehensive research on the security and privacy of LLM agents is highly needed. This survey aims to provide a comprehensive overview of the newly emerged privacy and security issues faced by LLM agents. We begin by introducing the fundamental knowledge of LLM agents, followed by a categorization and analysis of the threats. We then discuss the impacts of these threats on humans, environment, and other agents. Subsequently, we review existing defensive strategies, and finally explore future trends. Additionally, the survey incorporates diverse case studies to facilitate a more accessible understanding. By highlighting these critical security and privacy issues, the survey seeks to stimulate future research towards enhancing the security and privacy of LLM agents, thereby increasing their reliability and trustworthiness in future applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16648v3">FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted in the Findings of EMNLP, 2025
    </div>
    <details class="paper-abstract">
      The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06632v2">Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. Our code can be found on https://github.com/divelab/E2H-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20432v3">LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Strategic decision-making involves interactive reasoning where agents adapt their choices in response to others, yet existing evaluations of large language models (LLMs) often emphasize Nash Equilibrium (NE) approximation, overlooking the mechanisms driving their strategic choices. To bridge this gap, we introduce an evaluation framework grounded in behavioral game theory, disentangling reasoning capability from contextual effects. Testing 22 state-of-the-art LLMs, we find that GPT-o3-mini, GPT-o1, and DeepSeek-R1 dominate most games yet also demonstrate that the model scale alone does not determine performance. In terms of prompting enhancement, Chain-of-Thought (CoT) prompting is not universally effective, as it increases strategic reasoning only for models at certain levels while providing limited gains elsewhere. Additionally, we investigate the impact of encoded demographic features on the models, observing that certain assignments impact the decision-making pattern. For instance, GPT-4o shows stronger strategic reasoning with female traits than males, while Gemma assigns higher reasoning levels to heterosexual identities compared to other sexual orientations, indicating inherent biases. These findings underscore the need for ethical standards and contextual alignment to balance improved reasoning with fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10886v3">SLIP: Securing LLMs IP Using Weights Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently seen widespread adoption in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting substantial investments by their owners. The high cost of cloud-based deployment has spurred interest in running models on edge devices, but this risks exposing parameters to theft and unauthorized use. Existing approaches to protect model IP on the edge trade off practicality, accuracy, or deployment requirements. We introduce SLIP, a hybrid inference algorithm designed to protect edge-deployed models from theft. SLIP is, to our knowledge, the first hybrid protocol that is both practical for real-world applications and provably secure, while incurring zero accuracy degradation and minimal latency overhead. It partitions the model across two computing resources: one secure but expensive, and one cost-effective but vulnerable. Using matrix decomposition, the secure resource retains the most sensitive portion of the model's IP while performing only a small fraction of the computation; the vulnerable resource executes the remainder. The protocol includes security guarantees that prevent attackers from using the partition to infer the protected information. Finally, we present experimental results that demonstrate the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01390v2">Recognising, Anticipating, and Mitigating LLM Pollution of Online Behavioural Research</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Online behavioural research faces an emerging threat as participants increasingly turn to large language models (LLMs) for advice, translation, or task delegation: LLM Pollution. We identify three interacting variants through which LLM Pollution threatens the validity and integrity of online behavioural research. First, Partial LLM Mediation occurs when participants make selective use of LLMs for specific aspects of a task, such as translation or wording support, leading researchers to (mis)interpret LLM-shaped outputs as human ones. Second, Full LLM Delegation arises when agentic LLMs complete studies with little to no human oversight, undermining the central premise of human-subject research at a more foundational level. Third, LLM Spillover signifies human participants altering their behaviour as they begin to anticipate LLM presence in online studies, even when none are involved. While Partial Mediation and Full Delegation form a continuum of increasing automation, LLM Spillover reflects second-order reactivity effects. Together, these variants interact and generate cascading distortions that compromise sample authenticity, introduce biases that are difficult to detect post hoc, and ultimately undermine the epistemic grounding of online research on human cognition and behaviour. Crucially, the threat of LLM Pollution is already co-evolving with advances in generative AI, creating an escalating methodological arms race. To address this, we propose a multi-layered response spanning researcher practices, platform accountability, and community efforts. As the challenge evolves, coordinated adaptation will be essential to safeguard methodological integrity and preserve the validity of online behavioural research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14314v2">Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages fine-grained cross-model consistency to detect and mitigate hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves up to 9 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation on multiple datasets demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03343v3">What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train linear and non-linear probes on hidden states of open-weight LLMs to predict jailbreak success. Probes achieve strong in-distribution accuracy but transfer is attack-family-specific, revealing that different jailbreaks are supported by distinct internal mechanisms rather than a single universal direction. To establish causal relevance, we construct probe-guided latent interventions that systematically shift compliance in the predicted direction. Interventions derived from non-linear probes produce larger and more reliable effects than those from linear probes, indicating that features linked to jailbreak success are encoded non-linearly in prompt representations. Overall, the results surface heterogeneous, non-linear structure in jailbreak mechanisms and provide a prompt-side methodology for recovering and testing the features that drive jailbreak outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17336v3">PPMI: Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21189v2">Exploring the Hidden Capacity of LLMs for One-Step Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 accepted to EMNLP2025 main
    </div>
    <details class="paper-abstract">
      A recent study showed that large language models (LLMs) can reconstruct surprisingly long texts - up to thousands of tokens - via autoregressive generation from just one trained input embedding. In this work, we explore whether autoregressive decoding is essential for such reconstruction. We show that frozen LLMs can generate hundreds of accurate tokens in just one token-parallel forward pass, when provided with only two learned embeddings. This reveals a surprising and underexplored multi-token generation capability of autoregressive LLMs. We examine these embeddings and characterize the information they encode. We also empirically show that, although these representations are not unique for a given text, they form connected and local regions in embedding space - suggesting the potential to train a practical encoder. The existence of such representations hints that multi-token generation may be natively accessible in off-the-shelf LLMs via a learned input encoder, eliminating heavy retraining and helping to overcome the fundamental bottleneck of autoregressive decoding while reusing already-trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12872v2">KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Accepted for publication in NeurIPS2025. Code is available at \url{https://github.com/FastMAS/KVCOMM}
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems are increasingly adopted for complex language processing tasks that require communication and coordination among agents. However, these systems often suffer substantial overhead from repeated reprocessing of overlapping contexts across agents. In typical pipelines, once an agent receives a message from its predecessor, the full context-including prior turns-must be reprocessed from scratch, leading to inefficient processing. While key-value (KV) caching is an effective solution for avoiding redundant computation in single-agent settings where prefixes remain unchanged, it cannot be directly reused in multi-agent scenarios due to diverging prefixes introduced by agent-specific context extensions. We identify that the core challenge lies in the offset variance of KV-caches across agents. To address this, we propose KVCOMM, a training-free framework that enables efficient prefilling in multi-agent inference by reusing KV-caches and aligning cache offsets of overlapping contexts under diverse prefix contexts. KVCOMM estimates and adjusts KV-caches for shared content by referencing a pool of cached examples-termed anchors-that store observed cache deviations under varying prefixes. The anchor pool is maintained and updated online, allowing dynamic adaptation to distinct user requests and context structures. KVCOMM achieves over 70% reuse rate across diverse multi-agent workloads, including retrieval-augmented generation, math reasoning, and collaborative coding tasks, all without quality degradation. Particularly, when each fully-connected agent receives 1K input tokens with 512 prefix tokens and 512 output tokens under a five-agent setting, KVCOMM achieves up to 7.8x speedup compared to the standard prefill pipeline, reducing TTFT from ~430 ms to ~55 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24284v2">MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Preprint, Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on external tools to perform complex, realistic tasks, yet their ability to utilize the rapidly expanding Model Contextual Protocol (MCP) ecosystem remains limited. Existing MCP research covers few servers, depends on costly manual curation, and lacks training support, hindering progress toward real-world deployment. To overcome these limitations, we introduce MCP-Flow, an automated web-agent-driven pipeline for large-scale server discovery, data synthesis, and model training. MCP-Flow collects and filters data from 1166 servers and 11536 tools, producing 68733 high-quality instruction-function call pairs and 6439 trajectories, far exceeding prior work in scale and diversity. Extensive experiments demonstrate MCP-Flow's effectiveness in driving superior MCP tool selection, function-call generation, and enhanced agentic task performance. MCP-Flow thus provides a scalable foundation for advancing LLM agents' proficiency in real-world MCP environments. MCP-Flow is publicly available at \href{https://github.com/wwh0411/MCP-Flow}{https://github.com/wwh0411/MCP-Flow}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23874v3">From Stochasticity to Signal: A Bayesian Latent State Model for Reliable Measurement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate classification tasks in business, such as analyzing customer satisfaction from text. However, the inherent stochasticity of LLMs, in terms of their tendency to produce different outputs for the same input, creates a significant measurement error problem that is often neglected with a single round of output, or addressed with ad-hoc methods like majority voting. Such naive approaches fail to quantify uncertainty and can produce biased estimates of population-level metrics. In this paper, we propose a principled solution by reframing LLM variability as a statistical measurement error problem and introducing a Bayesian latent state model to address it. Our model treats the true classification (e.g., customer dissatisfaction) as an unobserved latent variable and the multiple LLM ratings as noisy measurements of this state. This framework allows for the simultaneous estimation of the LLM's false positive and false negative error rates, the underlying base rate of the phenomenon in the population, the posterior probability of the true state for each individual observation, and the causal impact of a business intervention, if any, on the latent state. Through simulation studies, we demonstrate that our model accurately recovers true parameters where naive methods fail. We conclude that this methodology provides a general and reliable framework for converting noisy, probabilistic outputs from LLMs into accurate and actionable insights for scientific and business applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17210v2">Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.
    </details>
</div>
