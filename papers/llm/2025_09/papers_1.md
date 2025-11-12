# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15849v2">The Impact of Language Mixing on Bilingual LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted at EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing-alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We show that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on MATH500. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by 2.92 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26625v1">Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Project page: https://junlinhan.github.io/projects/lsbs/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite being trained on text alone, surprisingly develop rich visual priors. These priors allow latent visual capabilities to be unlocked for vision tasks with a relatively small amount of multimodal data, and in some cases, to perform visual tasks without ever having seen an image. Through systematic analysis, we reveal that visual priors-the implicit, emergent knowledge about the visual world acquired during language pre-training-are composed of separable perception and reasoning priors with unique scaling trends and origins. We show that an LLM's latent visual reasoning ability is predominantly developed by pre-training on reasoning-centric data (e.g., code, math, academia) and scales progressively. This reasoning prior acquired from language pre-training is transferable and universally applicable to visual reasoning. In contrast, a perception prior emerges more diffusely from broad corpora, and perception ability is more sensitive to the vision encoder and visual instruction tuning data. In parallel, text describing the visual world proves crucial, though its performance impact saturates rapidly. Leveraging these insights, we propose a data-centric recipe for pre-training vision-aware LLMs and verify it in 1T token scale pre-training. Our findings are grounded in over 100 controlled experiments consuming 500,000 GPU-hours, spanning the full MLLM construction pipeline-from LLM pre-training to visual alignment and supervised multimodal fine-tuning-across five model scales, a wide range of data categories and mixtures, and multiple adaptation setups. Along with our main findings, we propose and investigate several hypotheses, and introduce the Multi-Level Existence Bench (MLE-Bench). Together, this work provides a new way of deliberately cultivating visual priors from language pre-training, paving the way for the next generation of multimodal LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26600v1">Deconstructing Self-Bias in LLM-generated Translation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) begin to saturate existing benchmarks, automated benchmark creation using LLMs (LLM as a benchmark) has emerged as a scalable alternative to slow and costly human curation. While these generated test sets have to potential to cheaply rank models, we demonstrate a critical flaw. LLM generated benchmarks systematically favor the model that created the benchmark, they exhibit self bias on low resource languages to English translation tasks. We show three key findings on automatic benchmarking of LLMs for translation: First, this bias originates from two sources: the generated test data (LLM as a testset) and the evaluation method (LLM as an evaluator), with their combination amplifying the effect. Second, self bias in LLM as a benchmark is heavily influenced by the model's generation capabilities in the source language. For instance, we observe more pronounced bias in into English translation, where the model's generation system is developed, than in out of English translation tasks. Third, we observe that low diversity in source text is one attribution to self bias. Our results suggest that improving the diversity of these generated source texts can mitigate some of the observed self bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23459v2">MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted to the 3rd Workshop on Regulatable ML at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising performance on tasks that require reasoning, such as text-to-SQL, code generation, and debugging. However, regulatory frameworks with strict privacy requirements constrain their integration into sensitive systems. State-of-the-art LLMs are also proprietary, costly, and resource-intensive, making local deployment impractical. Consequently, utilizing such LLMs often requires sharing data with third-party providers, raising privacy concerns and risking noncompliance with regulations. Although fine-tuned small language models (SLMs) can outperform LLMs on certain tasks and be deployed locally to mitigate privacy concerns, they underperform on more complex tasks such as text-to-SQL translation. In this work, we introduce MaskSQL, a text-to-SQL framework that utilizes abstraction as a privacy protection mechanism to mask sensitive information in LLM prompts. Unlike redaction, which removes content entirely, or generalization, which broadens tokens, abstraction retains essential information while discarding unnecessary details, striking an effective privacy-utility balance for the text-to-SQL task. Moreover, by providing mechanisms to control the privacy-utility tradeoff, MaskSQL facilitates adoption across a broader range of use cases. Our experimental results show that MaskSQL outperforms leading SLM-based text-to-SQL models and achieves performance approaching state-of-the-art LLM-based models, while preserving privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26578v1">Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have emerged as a promising approach to enhance the reasoning capabilities of large language models (LLMs) by guiding their step-by-step reasoning toward a final answer. However, existing PRMs either treat each reasoning step in isolation, failing to capture inter-step dependencies, or struggle to align process rewards with the final outcome. Consequently, the reward signal fails to respect temporal causality in sequential reasoning and faces ambiguous credit assignment. These limitations make downstream models vulnerable to reward hacking and lead to suboptimal performance. In this work, we propose Conditional Reward Modeling (CRM) that frames LLM reasoning as a temporal process leading to a correct answer. The reward of each reasoning step is not only conditioned on the preceding steps but also explicitly linked to the final outcome of the reasoning trajectory. By enforcing conditional probability rules, our design captures the causal relationships among reasoning steps, with the link to the outcome allowing precise attribution of each intermediate step, thereby resolving credit assignment ambiguity. Further, through this consistent probabilistic modeling, the rewards produced by CRM enable more reliable cross-sample comparison. Experiments across Best-of-N sampling, beam search and reinforcement learning demonstrate that CRM consistently outperforms existing reward models, offering a principled framework for enhancing LLM reasoning. In particular, CRM is more robust to reward hacking and delivers stable downstream improvements without relying on verifiable rewards derived from ground truth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03015v2">AuDeRe: Automated Strategy Decision and Realization in Robot Planning and Control via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 8 pages, 14 figures, submitted to the 2026 American Control Conference
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown significant promise in various domains, especially robotics. However, most prior LLM-based work in robotic applications either directly predicts waypoints or applies LLMs within fixed tool integration frameworks, offering limited flexibility in exploring and configuring solutions best suited to different tasks. In this work, we propose a framework that leverages LLMs to select appropriate planning and control strategies based on task descriptions, environmental constraints, and system dynamics. These strategies are then executed by calling the available comprehensive planning and control APIs. Our approach employs iterative LLM-based reasoning with performance feedback to refine the algorithm selection. We validate our approach through extensive experiments across tasks of varying complexity, from simple tracking to complex planning scenarios involving spatiotemporal constraints. The results demonstrate that using LLMs to determine planning and control strategies from natural language descriptions significantly enhances robotic autonomy while reducing the need for extensive manual tuning and expert knowledge. Furthermore, our framework maintains generalizability across different tasks and notably outperforms baseline methods that rely on LLMs for direct trajectory, control sequence, or code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26553v1">Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      As language models gain access to external tools via structured function calls, they become increasingly more capable of solving complex, multi-step tasks. However, existing benchmarks for tool-augmented language models (TaLMs) provide insufficient control over factors such as the number of functions accessible, task complexity, and input size, and remain vulnerable to data contamination. We present FuncBenchGen, a unified, contamination-free framework that evaluates TaLMs by generating synthetic multi-step tool-use tasks. The key idea is to cast tool use as traversal over a hidden function-dependency DAG where nodes are function calls and an edge between nodes represents one function consuming the output of another. Given a set of external function schemas, initial variable values, and a target variable, models must compose the correct call sequence to compute the target variable. FuncBenchGen allows users to precisely control task difficulty (e.g., graph size, dependency depth, and distractor functions) while avoiding data leakage. We apply our FuncBenchGen framework to evaluate seven LLMs on tool use tasks of varying difficulty. Reasoning-optimized models consistently outperform general-purpose models with GPT-5 significantly outperforming other models. Performance declines sharply as dependency depth increases. Furthermore, connected irrelevant functions prove especially difficult to handle. We find that strong models often make syntactically valid function calls but propagate incorrect or stale argument values across steps, revealing brittle state tracking by LLMs in multi-turn tool use. Motivated by this observation, we introduce a simple mitigation strategy that explicitly restates prior variable values to the agent at each step. Surprisingly, this lightweight change yields substantial gains across models. e.g., yielding a success rate improvement from 62.5% to 81.3% for GPT-5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26546v1">Towards Verified Code Reasoning by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 43 pages
    </div>
    <details class="paper-abstract">
      While LLM-based agents are able to tackle a wide variety of code reasoning questions, the answers are not always correct. This prevents the agent from being useful in situations where high precision is desired: (1) helping a software engineer understand a new code base, (2) helping a software engineer during code review sessions, and (3) ensuring that the code generated by an automated code generation system meets certain requirements (e.g. fixes a bug, improves readability, implements a feature). As a result of this lack of trustworthiness, the agent's answers need to be manually verified before they can be trusted. Manually confirming responses from a code reasoning agent requires human effort and can result in slower developer productivity, which weakens the assistance benefits of the agent. In this paper, we describe a method to automatically validate the answers provided by a code reasoning agent by verifying its reasoning steps. At a very high level, the method consists of extracting a formal representation of the agent's response and, subsequently, using formal verification and program analysis tools to verify the agent's reasoning steps. We applied this approach to a benchmark set of 20 uninitialized variable errors detected by sanitizers and 20 program equivalence queries. For the uninitialized variable errors, the formal verification step was able to validate the agent's reasoning on 13/20 examples, and for the program equivalence queries, the formal verification step successfully caught 6/8 incorrect judgments made by the agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26514v1">BatonVoice: An Operationalist Framework for Enhancing Controllable Speech Synthesis with Linguistic Intelligence from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) is reshaping multimodel models, with speech synthesis being a prominent application. However, existing approaches often underutilize the linguistic intelligence of these models, typically failing to leverage their powerful instruction-following capabilities. This limitation hinders the model's ability to follow text instructions for controllable Text-to-Speech~(TTS). To address this, we propose a new paradigm inspired by ``operationalism'' that decouples instruction understanding from speech generation. We introduce BatonVoice, a framework where an LLM acts as a ``conductor'', understanding user instructions and generating a textual ``plan'' -- explicit vocal features (e.g., pitch, energy). A separate TTS model, the ``orchestra'', then generates the speech from these features. To realize this component, we develop BatonTTS, a TTS model trained specifically for this task. Our experiments demonstrate that BatonVoice achieves strong performance in controllable and emotional speech synthesis, outperforming strong open- and closed-source baselines. Notably, our approach enables remarkable zero-shot cross-lingual generalization, accurately applying feature control abilities to languages unseen during post-training. This demonstrates that objectifying speech into textual vocal features can more effectively unlock the linguistic intelligence of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26490v1">VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 The code, dataset, and leaderboard are available at https://vitabench.github.io/
    </div>
    <details class="paper-abstract">
      As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at https://vitabench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26432v1">AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) are gaining attention for their inherent capacity for parallel decoding, offering a compelling alternative to autoregressive LLMs. Among various decoding strategies, blockwise semi-autoregressive (semi-AR) approaches are widely adopted due to their natural support for KV caching and their favorable accuracy-speed trade-off. However, this paper identifies two fundamental limitations in the conventional semi-AR decoding approach that applies a fixed block size: i) late decoding overhead, where the unmasking of high-confidence tokens outside the current block is unnecessarily delayed, and ii) premature decoding error, where low-confidence tokens inside the current block are committed too early, leading to incorrect tokens. This paper presents the first systematic investigation challenging the fixed block size assumption in semi-AR decoding. Through a statistical analysis of confidence dynamics during the denoising process, we identify a volatility band (VB) region during dLLM decoding, which encodes local semantic structure and can be used to guide adaptive block sizing. Leveraging these insights, we introduce AdaBlock-dLLM, a training-free, plug-and-play scheduler that adaptively aligns block boundaries with semantic steps by adjusting block size during runtime. Extensive experiments across diverse benchmarks show that AdaBlock-dLLM achieves up to 5.3% accuracy improvement under the same throughput budget. Beyond inference-time optimization, we hope our semantics-aware adaptive scheduling approach and confidence-based analysis will inspire future training strategies for dLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16293v2">Robust LLM Training Infrastructure at ByteDance</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The training scale of large language models (LLMs) has reached tens of thousands of GPUs and is still continuously expanding, enabling faster learning of larger models. Accompanying the expansion of the resource scale is the prevalence of failures (CUDA error, NaN values, job hang, etc.), which poses significant challenges to training stability. Any large-scale LLM training infrastructure should strive for minimal training interruption, efficient fault diagnosis, and effective failure tolerance to enable highly efficient continuous training. This paper presents ByteRobust, a large-scale GPU infrastructure management system tailored for robust and stable training of LLMs. It exploits the uniqueness of LLM training process and gives top priorities to detecting and recovering failures in a routine manner. Leveraging parallelisms and characteristics of LLM training, ByteRobust enables high-capacity fault tolerance, prompt fault demarcation, and localization with an effective data-driven approach, comprehensively ensuring continuous and efficient training of LLM tasks. ByteRobust is deployed on a production GPU platform with over 200,000 GPUs and achieves 97% ETTR for a three-month training job on 9,600 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07861v2">Scalable LLM Math Reasoning Acceleration with Low-rank Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a resource-efficient distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>16% time-to-next-token reduction) while encouraging response brevity (up to 8.5% fewer tokens).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12379v2">GraphCogent: Mitigating LLMs' Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promising performance on small-scale graph reasoning tasks but fail when handling real-world graphs with complex queries. This phenomenon arises from LLMs' working memory constraints, which result in their inability to retain long-range graph topology over extended contexts while sustaining coherent multi-step reasoning. However, real-world graphs are often structurally complex, such as Web, Transportation, Social, and Citation networks. To address these limitations, we propose GraphCogent, a collaborative agent framework inspired by human Working Memory Model that decomposes graph reasoning into specialized cognitive processes: sense, buffer, and execute. The framework consists of three modules: Sensory Module standardizes diverse graph text representations via subgraph sampling, Buffer Module integrates and indexes graph data across multiple formats, and Execution Module combines tool calling and tool creation for efficient reasoning. We also introduce Graph4real, a comprehensive benchmark that contains four domains of real-world graphs (Web, Transportation, Social, and Citation) to evaluate LLMs' graph reasoning capabilities. Our Graph4real covers 21 different graph reasoning tasks, categorized into three types (Structural Querying, Algorithmic Reasoning, and Predictive Modeling tasks), with graph scales up to 10 times larger than existing benchmarks. Experiments show that Llama3.1-8B based GraphCogent achieves a 50% improvement over massive-scale LLMs like DeepSeek-R1 (671B). Compared to state-of-the-art agent-based baseline, our framework outperforms by 20% in accuracy while reducing token usage by 80% for in-toolset tasks and 30% for out-toolset tasks. Code will be available after review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26354v1">Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have enabled a new class of self-evolving agents that autonomously improve through interaction with the environment, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study the case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as Misevolution. To provide a systematic investigation, we evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (e.g., Gemini-2.5-Pro). Different emergent risks are observed in the self-evolutionary process, such as the degradation of safety alignment after memory accumulation, or the unintended introduction of vulnerabilities in tool creation and reuse. To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents. Our code and data are available at https://github.com/ShaoShuai0605/Misevolution . Warning: this paper includes examples that may be offensive or harmful in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26351v1">LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Submitted to GenAI4Health@NeurIPS 2025. This is the first version of the LLM-assisted emergency triage benchmark dataset and baseline models. Authors: Joshua Sebastian, Karma Tobden, KMA Solaiman
    </div>
    <details class="paper-abstract">
      Research on emergency and mass casualty incident (MCI) triage has been limited by the absence of openly usable, reproducible benchmarks. Yet these scenarios demand rapid identification of the patients most in need, where accurate deterioration prediction can guide timely interventions. While the MIMIC-IV-ED database is openly available to credentialed researchers, transforming it into a triage-focused benchmark requires extensive preprocessing, feature harmonization, and schema alignment -- barriers that restrict accessibility to only highly technical users. We address these gaps by first introducing an open, LLM-assisted emergency triage benchmark for deterioration prediction (ICU transfer, in-hospital mortality). The benchmark then defines two regimes: (i) a hospital-rich setting with vitals, labs, notes, chief complaints, and structured observations, and (ii) an MCI-like field simulation limited to vitals, observations, and notes. Large language models (LLMs) contributed directly to dataset construction by (i) harmonizing noisy fields such as AVPU and breathing devices, (ii) prioritizing clinically relevant vitals and labs, and (iii) guiding schema alignment and efficient merging of disparate tables. We further provide baseline models and SHAP-based interpretability analyses, illustrating predictive gaps between regimes and the features most critical for triage. Together, these contributions make triage prediction research more reproducible and accessible -- a step toward dataset democratization in clinical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20295v3">SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The common approach to communicate a large language model's (LLM) uncertainty is to add a percentage number or a hedging word to its response. But is this all we can do? Instead of generating a single answer and then hedging it, an LLM that is fully transparent to the user needs to be able to reflect on its internal belief distribution and output a summary of all options it deems possible, and how likely they are. To test whether LLMs possess this capability, we develop the SelfReflect metric, an information-theoretic distance between a given summary and a distribution over answers. In interventional and human studies, we find that SelfReflect indicates even slight deviations, yielding a fine measure of faithfulness between a summary string and an LLM's actual internal distribution over answers. With SelfReflect, we make a resounding negative observation: modern LLMs are, across the board, incapable of revealing what they are uncertain about, neither through reasoning, nor chains-of-thoughts, nor explicit finetuning. However, we do find that LLMs are able to generate faithful summaries of their uncertainties if we help them by sampling multiple outputs and feeding them back into the context. This simple approach shines a light at the universal way of communicating LLM uncertainties whose future development the SelfReflect score enables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26328v1">Fast-dLLM v2: Efficient Block-Diffusion LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Autoregressive (AR) large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks, yet their inherent sequential decoding limits inference efficiency. In this work, we propose Fast-dLLM v2, a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained AR models into dLLMs for parallel text generation, requiring only approximately 1B tokens of fine-tuning. This represents a 500x reduction in training data compared to full-attention diffusion LLMs such as Dream (580B tokens), while preserving the original model's performance. Our approach introduces a novel training recipe that combines a block diffusion mechanism with a complementary attention mask, enabling blockwise bidirectional context modeling without sacrificing AR training objectives. To further accelerate decoding, we design a hierarchical caching mechanism: a block-level cache that stores historical context representations across blocks, and a sub-block cache that enables efficient parallel generation within partially decoded blocks. Coupled with our parallel decoding pipeline, Fast-dLLM v2 achieves up to 2.5x speedup over standard AR decoding without compromising generation quality. Extensive experiments across diverse benchmarks demonstrate that Fast-dLLM v2 matches or surpasses AR baselines in accuracy, while delivering state-of-the-art efficiency among dLLMs - marking a significant step toward the practical deployment of fast and accurate LLMs. Code and model will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26313v1">One-Token Rollout: Guiding Supervised Fine-Tuning of LLMs with Policy Gradient</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is the predominant method for adapting large language models (LLMs), yet it often struggles with generalization compared to reinforcement learning (RL). In this work, we posit that this performance disparity stems not just from the loss function, but from a more fundamental difference: SFT learns from a fixed, pre-collected dataset, whereas RL utilizes on-policy data sampled from the current policy. Building on this hypothesis, we introduce one-token rollout (OTR), a novel fine-tuning algorithm that guides SFT with the policy gradient method. OTR reframes the autoregressive learning process by treating each token generation as a single-step reinforcement learning trajectory. At each step, it performs a Monte Carlo ``rollout'' by sampling multiple candidate tokens from the current policy's distribution. The ground-truth token from the supervised data is then used to provide a reward signal to these samples. Guided by policy gradient, our algorithm repurposes static, off-policy supervised data into a dynamic, on-policy signal at the token level, capturing the generalization benefits of on-policy learning while bypassing the costly overhead of full sentence generation. Through extensive experiments on a diverse suite of challenging benchmarks spanning mathematical reasoning, code generation, and general domain reasoning, we demonstrate that OTR consistently outperforms standard SFT. Our findings establish OTR as a powerful and practical alternative for fine-tuning LLMs and provide compelling evidence that the on-policy nature of data is a critical driver of generalization, offering a promising new direction for fine-tuning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26306v1">Interactive Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 The code will be released later
    </div>
    <details class="paper-abstract">
      Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14043v3">Beyond Next Token Probabilities: Learnable, Fast Detection of Hallucinations and Data Contamination on LLM Output Distributions</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The automated detection of hallucinations and training data contamination is pivotal to the safe deployment of Large Language Models (LLMs). These tasks are particularly challenging in settings where no access to model internals is available. Current approaches in this setup typically leverage only the probabilities of actual tokens in the text, relying on simple task-specific heuristics. Crucially, they overlook the information contained in the full sequence of next-token probability distributions. We propose to go beyond hand-crafted decision rules by learning directly from the complete observable output of LLMs -- consisting not only of next-token probabilities, but also the full sequence of next-token distributions. We refer to this as the LLM Output Signature (LOS), and treat it as a reference data type for detecting hallucinations and data contamination. To that end, we introduce LOS-Net, a lightweight attention-based architecture trained on an efficient encoding of the LOS, which can provably approximate a broad class of existing techniques for both tasks. Empirically, LOS-Net achieves superior performance across diverse benchmarks and LLMs, while maintaining extremely low detection latency. Furthermore, it demonstrates promising transfer capabilities across datasets and LLMs. Full code is available at https://github.com/BarSGuy/Beyond-next-token-probabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26246v1">SlimPack: Fine-Grained Asymmetric Packing for Balanced and Efficient Variable-Length LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The efficient distributed training of Large Language Models (LLMs) is severely hampered by the extreme variance in context lengths. This data heterogeneity, amplified by conventional packing strategies and asymmetric forward-backward costs, leads to critical inefficiencies such as cascading workload imbalances and severe hardware underutilization. Existing solutions attempt to mitigate these challenges, but often at the expense of memory or communication efficiency. To address these challenges, we introduce SlimPack, a framework that fundamentally rethinks data packing and scheduling by decomposing samples into fine-grained slices. This slice-level decomposition immediately mitigates critical memory and communication bottlenecks by transforming large, volatile workloads into a stream of smaller, manageable units. This flexibility is then harnessed for our core innovation, Asymmetric Partitioning, which assembles balanced scheduling units uniquely optimized for the different demands of the forward and backward passes. Orchestrated by a two-phase solver and a high-fidelity simulator, SlimPack holistically resolves imbalances across all parallel dimensions. Extensive experiments demonstrate that SlimPack achieves up to a $2.8\times$ training throughput improvement over baselines, breaking the conventional trade-off by delivering both superior balance and high resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11843v4">Preemptive Detection and Correction of Misaligned Actions in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Deploying LLM-based agents in real-life applications often faces a critical challenge: the misalignment between agents' behavior and user intent. Such misalignment may lead agents to unintentionally execute critical actions that carry negative outcomes (e.g., accidentally triggering a "buy-now" in web shopping), resulting in undesirable or even irreversible consequences. Although addressing these issues is crucial, the preemptive detection and correction of misaligned actions remains relatively underexplored. To fill this gap, we introduce InferAct, a novel approach that leverages the belief reasoning ability of LLMs, grounded in Theory-of-Mind, to detect misaligned actions before execution. Once the misalignment is detected, InferAct alerts users for timely correction, preventing adverse outcomes and enhancing the reliability of LLM agents' decision-making processes. Experiments on three widely used tasks demonstrate that InferAct achieves up to 20% improvements on Marco-F1 against baselines in misaligned action detection. An in-depth evaluation of misalignment correction further highlights InferAct's effectiveness in improving agent alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20230v3">Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26201v1">LLM Agents for Knowledge Discovery in Atomic Layer Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted submission to the AI4MAT workshop@NEURIPS 2025. As submitted, except author names added
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have garnered significant attention for several years now. Recently, their use as independently reasoning agents has been proposed. In this work, we test the potential of such agents for knowledge discovery in materials science. We repurpose LangGraph's tool functionality to supply agents with a black box function to interrogate. In contrast to process optimization or performing specific, user-defined tasks, knowledge discovery consists of freely exploring the system, posing and verifying statements about the behavior of this black box, with the sole objective of generating and verifying generalizable statements. We provide proof of concept for this approach through a children's parlor game, demonstrating the role of trial-and-error and persistence in knowledge discovery, and the strong path-dependence of results. We then apply the same strategy to show that LLM agents can explore, discover, and exploit diverse chemical interactions in an advanced Atomic Layer Processing reactor simulation using intentionally limited probe capabilities without explicit instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26200v1">Toward an Unbiased Collective Memory for Efficient LLM-Based Agentic 6G Cross-Domain Management</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      This paper introduces a novel framework for proactive cross-domain resource orchestration in 6G RAN-Edge networks, featuring large language model (LLM)-augmented agents. The system comprises specialized RAN (energy efficiency) and Edge (latency assurance) agents that engage in iterative negotiation, supported by advanced reasoning and planning capabilities. Agents dynamically interact with a digital twin (DT) to test their proposals and leverage a long-term collective memory where their joint successful and failed agreements along with the related network contexts are distilled into strategies to either follow or avoid and subsequently stored. Given that agents are subject to a plethora of cognitive distortions when retrieving those past experiences -- such as primacy, recency, confirmation and availability biases -- we propose in this work a novel unbiased memory design (A reusable mockup version of the unbiased memory source code is available for non-commercial use at https://github.com/HatimChergui/unbiased-collective-memory). featuring (i) semantic retrieval of past strategies via Jaccard similarity; (ii) learning from failures through amplified weighting of SLA violations and mandatory inclusion of failed negotiation cases to mitigate confirmation bias; (iii) diversity enforcement to minimize availability bias and (iv) recency and primacy weighting with slow decay to counteract temporal biases. Evaluation results showcase the impact of existing biases and how the unbiased memory allows to tackle them by learning from both successful and failed strategies, either present or old, resulting in $\times 4.5$ and $\times 3.5$ reductions of unresolved negotiations compared to non-memory and vanilla memory baselines, respectively, while totally mitigating SLA violations as well as improving latency and energy saving distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17148v2">LLM Agents for Interactive Exploration of Historical Cadastre Data: Framework and Application to Venice</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted in Cambridge press - Computational Humanities Research 2025
    </div>
    <details class="paper-abstract">
      Cadastral data reveal key information about the historical organization of cities but are often non-standardized due to diverse formats and human annotations, complicating large-scale analysis. We explore as a case study Venice's urban history during the critical period from 1740 to 1808, capturing the transition following the fall of the ancient Republic and the Ancien R\'egime. This era's complex cadastral data, marked by its volume and lack of uniform structure, presents unique challenges that our approach adeptly navigates, enabling us to generate spatial queries that bridge past and present urban landscapes. We present a text-to-programs framework that leverages Large Language Models (\llms) to process natural language queries as executable code for analyzing historical cadastral records. Our methodology implements two complementary techniques: a SQL agent for handling structured queries about specific cadastral information, and a coding agent for complex analytical operations requiring custom data manipulation. We propose a taxonomy that classifies historical research questions based on their complexity and analytical requirements, mapping them to the most appropriate technical approach. This framework is supported by an investigation into the execution consistency of the system, alongside a qualitative analysis of the answers it produces. By ensuring interpretability and minimizing hallucination through verifiable program outputs, we demonstrate the system's effectiveness in reconstructing past population information, property features, and spatiotemporal comparisons in Venice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26189v1">VietBinoculars: A Zero-Shot Approach for Detecting Vietnamese LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      The rapid development research of Large Language Models (LLMs) based on transformer architectures raises key challenges, one of them being the task of distinguishing between human-written text and LLM-generated text. As LLM-generated textual content, becomes increasingly complex over time, and resembles human writing, traditional detection methods are proving less effective, especially as the number and diversity of LLMs continue to grow with new models and versions being released at a rapid pace. This study proposes VietBinoculars, an adaptation of the Binoculars method with optimized global thresholds, to enhance the detection of Vietnamese LLM-generated text. We have constructed new Vietnamese AI-generated datasets to determine the optimal thresholds for VietBinoculars and to enable benchmarking. The results from our experiments show results show that VietBinoculars achieves over 99\% in all two domains of accuracy, F1-score, and AUC on multiple out-of-domain datasets. It outperforms the original Binoculars model, traditional detection methods, and other state-of-the-art approaches, including commercial tools such as ZeroGPT and DetectGPT, especially under specially modified prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26184v1">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 ECIR 2025 demo format
    </div>
    <details class="paper-abstract">
      Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recent ARGUE framework for report generation evaluation. We present analysis of Auto-ARGUE on the report generation pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26182v1">Parallax: Efficient LLM Inference Service over Decentralized Environment</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Deploying a large language model (LLM) inference service remains costly because centralized serving depends on specialized GPU clusters and high-bandwidth interconnects in datacenters. An appealing alternative is to leverage collaborative decentralized GPU pools. However, heterogeneity in GPU and limited interconnected network bandwidth, along with potentially dynamic availability, make efficient scheduling the central challenge in this scenario. In this paper, we present Parallax, a decentralized LLM serving system that turns a pool of heterogeneous GPUs into an efficient inference platform via a two-phase scheduler. Parallax decomposes planning into (i) model allocation, which places layers of each replica across diverse GPUs to jointly optimize latency and throughput under memory and link-bandwidth constraints, and (ii) request-time GPU pipeline selection, which stitches layers from different replicas into end-to-end execution chains that balance load and adapt to current conditions. We implement Parallax and evaluate it on open-source LLMs deployed over real volunteer nodes. Parallax consistently reduces latency and increases throughput relative to decentralized baselines, demonstrating that principled scheduling can make volunteer compute a practical, affordable substrate for LLM inference. Github Repo at: https://github.com/GradientHQ/parallax.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26167v1">'Too much alignment; not enough culture': Re-balancing cultural alignment practices in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 8 pages, no figures
    </div>
    <details class="paper-abstract">
      While cultural alignment has increasingly become a focal point within AI research, current approaches relying predominantly on quantitative benchmarks and simplistic proxies fail to capture the deeply nuanced and context-dependent nature of human cultures. Existing alignment practices typically reduce culture to static demographic categories or superficial cultural facts, thereby sidestepping critical questions about what it truly means to be culturally aligned. This paper argues for a fundamental shift towards integrating interpretive qualitative approaches drawn from social sciences into AI alignment practices, specifically in the context of Large Language Models (LLMs). Drawing inspiration from Clifford Geertz's concept of "thick description," we propose that AI systems must produce outputs that reflect deeper cultural meanings--what we term "thick outputs"-grounded firmly in user-provided context and intent. We outline three necessary conditions for successful cultural alignment: sufficiently scoped cultural representations, the capacity for nuanced outputs, and the anchoring of outputs in the cultural contexts implied within prompts. Finally, we call for cross-disciplinary collaboration and the adoption of qualitative, ethnographic evaluation methods as vital steps toward developing AI systems that are genuinely culturally sensitive, ethically responsible, and reflective of human complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26124v1">Vocabulary Customization for Efficient Domain-Specific LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted at NEURIPS 2025 CCFM Workshop
    </div>
    <details class="paper-abstract">
      When using an LLM to process text outside the training domain(s), an often overlooked factor is vocabulary mismatch, where the general-domain tokenizer fails to capture frequent domain-specific terms, leading to higher token fertility and thus a decrease in processing speed due to suboptimal sub-word splits. We address this limitation by augmenting the pretrained vocabulary with a set of domain-specific tokens. To this end, we design an algorithm that extends an existing tokenizer while guaranteeing it never decreases tokenization efficiency: every input sequence is segmented into at most the same number of tokens as before. Evaluated on real-world e-Commerce use-cases, the augmented tokenizer significantly shortens input sequences by up to 20% and reduces inference latency on downstream tasks while preserving predictive quality. We further analyze secondary effects, such as the impact on forward pass speed and the rate at which the model adopts the newly introduced tokens, to illustrate the broader benefits of vocabulary adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26100v1">SafeEvalAgent: Toward Agentic and Self-Evolving Safety Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The rapid integration of Large Language Models (LLMs) into high-stakes domains necessitates reliable safety and compliance evaluation. However, existing static benchmarks are ill-equipped to address the dynamic nature of AI risks and evolving regulations, creating a critical safety gap. This paper introduces a new paradigm of agentic safety evaluation, reframing evaluation as a continuous and self-evolving process rather than a one-time audit. We then propose a novel multi-agent framework SafeEvalAgent, which autonomously ingests unstructured policy documents to generate and perpetually evolve a comprehensive safety benchmark. SafeEvalAgent leverages a synergistic pipeline of specialized agents and incorporates a Self-evolving Evaluation loop, where the system learns from evaluation results to craft progressively more sophisticated and targeted test cases. Our experiments demonstrate the effectiveness of SafeEvalAgent, showing a consistent decline in model safety as the evaluation hardens. For instance, GPT-5's safety rate on the EU AI Act drops from 72.50% to 36.36% over successive iterations. These findings reveal the limitations of static assessments and highlight our framework's ability to uncover deep vulnerabilities missed by traditional methods, underscoring the urgent need for dynamic evaluation ecosystems to ensure the safe and responsible deployment of advanced AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23058v2">Risk Profiling and Modulation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for decision-making tasks under uncertainty; however, their risk profiles and how they are influenced by prompting and alignment methods remain underexplored. Existing studies have primarily examined personality prompting or multi-agent interactions, leaving open the question of how post-training influences the risk behavior of LLMs. In this work, we propose a new pipeline for eliciting, steering, and modulating LLMs' risk profiles, drawing on tools from behavioral economics and finance. Using utility-theoretic models, we compare pre-trained, instruction-tuned, and RLHF-aligned LLMs, and find that while instruction-tuned models exhibit behaviors consistent with some standard utility formulations, pre-trained and RLHF-aligned models deviate more from any utility models fitted. We further evaluate modulation strategies, including prompt engineering, in-context learning, and post-training, and show that post-training provides the most stable and effective modulation of risk preference. Our findings provide insights into the risk profiles of different classes and stages of LLMs and demonstrate how post-training modulates these profiles, laying the groundwork for future research on behavioral alignment and risk-aware LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26072v1">The Silent Judge: Unacknowledged Shortcut Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 9 Pages, 5 Tables, 1 Figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as automatic judges to evaluate system outputs in tasks such as summarization, dialogue, and creative writing. A faithful judge should base its verdicts solely on response quality and explicitly acknowledge the factors shaping its decision. We show that current LLM judges fail on both counts by relying on shortcuts introduced in the prompt. Our study uses two evaluation datasets: ELI5, a benchmark for long-form question answering, and LitBench, a recent benchmark for creative writing. Both datasets provide pairwise comparisons, where the evaluator must choose which of two responses is better. From each dataset we construct 100 pairwise judgment tasks and employ two widely used models, GPT-4o and Gemini-2.5-Flash, as evaluators in the role of LLM-as-a-judge. For each pair, we assign superficial cues to the responses, provenance cues indicating source identity (Human, Expert, LLM, or Unknown) and recency cues indicating temporal origin (Old, 1950 vs. New, 2025), while keeping the rest of the prompt fixed. Results reveal consistent verdict shifts: both models exhibit a strong recency bias, systematically favoring new responses over old, as well as a clear provenance hierarchy (Expert > Human > LLM > Unknown). These biases are especially pronounced in GPT-4o and in the more subjective and open-ended LitBench domain. Crucially, cue acknowledgment is rare: justifications almost never reference the injected cues, instead rationalizing decisions in terms of content qualities. These findings demonstrate that current LLM-as-a-judge systems are shortcut-prone and unfaithful, undermining their reliability as evaluators in both research and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06346v2">Ban&Pick: Ehancing Performance and Efficiency of MoE-LLMs via Smarter Routing</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 21 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency at inference. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban further dynamically prunes redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban\&Pick delivers free performance gains and inference acceleration without retraining or architectural changes. For instance, on Qwen3-30B-A3B, it improves accuracy from 80.67 to 84.66 on AIME2024 and from 65.66 to 68.18 on GPQA-Diamond, while accelerating inference by 1.25x under the vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22725v2">A Meta-Analysis of LLM Effects on Students across Qualification, Socialisation, and Subjectification</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly positioned as solutions for education, yet evaluations often reduce their impact to narrow performance metrics. This paper reframes the question by asking "what kind of impact should LLMs have in education?" Drawing on Biesta's tripartite account of good education: qualification, socialisation, and subjectification, we present a meta-analysis of 133 experimental and quasi-experimental studies (k = 188). Overall, the impact of LLMs on student learning is positive but uneven. Strong effects emerge in qualification, particularly when LLMs function as tutors in sustained interventions. Socialisation outcomes appear more variable, concentrated in sustained, reflective interventions. Subjectification, linked to autonomy and learner development, remains fragile, with improvements confined to small-scale, long-term studies. This purpose-level view highlights design as the decisive factor: without scaffolds for participation and agency, LLMs privilege what is easiest to measure while neglecting broader aims of education. For HCI and education, the issue is not just whether LLMs work, but what futures they enable or foreclose.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26041v1">Unspoken Hints: Accuracy Without Acknowledgement in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 5 Pages, 4 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on chain-of-thought (CoT) prompting to solve mathematical and logical reasoning tasks. Yet, a central question remains: to what extent are these generated rationales \emph{faithful} to the underlying computations, rather than post-hoc narratives shaped by hints that function as answer shortcuts embedded in the prompt? Following prior work on hinted vs.\ unhinted prompting, we present a systematic study of CoT faithfulness under controlled hint manipulations. Our experimental design spans four datasets (AIME, GSM-Hard, MATH-500, UniADILR), two state-of-the-art models (GPT-4o and Gemini-2-Flash), and a structured set of hint conditions varying in correctness (correct and incorrect), presentation style (sycophancy and data leak), and complexity (raw answers, two-operator expressions, four-operator expressions). We evaluate both task accuracy and whether hints are explicitly acknowledged in the reasoning. Our results reveal three key findings. First, correct hints substantially improve accuracy, especially on harder benchmarks and logical reasoning, while incorrect hints sharply reduce accuracy in tasks with lower baseline competence. Second, acknowledgement of hints is highly uneven: equation-based hints are frequently referenced, whereas raw hints are often adopted silently, indicating that more complex hints push models toward verbalizing their reliance in the reasoning process. Third, presentation style matters: sycophancy prompts encourage overt acknowledgement, while leak-style prompts increase accuracy but promote hidden reliance. This may reflect RLHF-related effects, as sycophancy exploits the human-pleasing side and data leak triggers the self-censoring side. Together, these results demonstrate that LLM reasoning is systematically shaped by shortcuts in ways that obscure faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00404v4">Metis: Training LLMs with FP4 Quantization</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      This work identifies anisotropy in the singular value spectra of parameters, activations, and gradients as the fundamental barrier to low-bit training of large language models (LLMs). These spectra are dominated by a small fraction of large singular values, inducing wide numerical ranges that cause quantization bias and severe spectral distortion, ultimately degrading training performance. This work presents Metis, a spectral-domain quantization framework that partitions anisotropic spectra into narrower sub-distributions for independent quantization, thereby reducing errors and preserving spectral structure. To minimize overhead, Metis leverages two key properties of the dominant spectral subspace: preservation via sparsely random sampling and preservation via random projection, reducing decomposition cost to a negligible level. On LLaMA-3 8B trained with 100B tokens, Metis enables robust W4A4G4 training with FP4 quantization of weights, activations, and gradients, yielding only a 0.4% training loss gap and a 0.1% degradation in downstream accuracy relative to BF16. Beyond matching BF16 fidelity, Metis also surpasses our implementation of Nvidia's recently announced (yet to be publicly released) FP4 recipe, consistently achieving lower loss and higher downstream accuracy while incurring significantly lower computational overhead. The code implementation for Metis is available at: https://anonymous.4open.science/r/Metis-quantization-644B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25992v1">MHINDR -- a DSM5 based mental health diagnosis and recommendation framework using LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 7 pages, 1 figure, 4 tables
    </div>
    <details class="paper-abstract">
      Mental health forums offer valuable insights into psychological issues, stressors, and potential solutions. We propose MHINDR, a large language model (LLM) based framework integrated with DSM-5 criteria to analyze user-generated text, dignose mental health conditions, and generate personalized interventions and insights for mental health practitioners. Our approach emphasizes on the extraction of temporal information for accurate diagnosis and symptom progression tracking, together with psychological features to create comprehensive mental health summaries of users. The framework delivers scalable, customizable, and data-driven therapeutic recommendations, adaptable to diverse clinical contexts, patient needs, and workplace well-being programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25987v1">R-Log: Incentivizing Log Analysis Capability in LLMs via Reasoning-based Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The growing complexity of log data in modern software systems has prompted the use of Large Language Models (LLMs) for automated log analysis. Current approaches typically rely on direct supervised fine-tuning (SFT) on log-label pairs. However, this exacerbates the domain discrepancy between general-purpose LLMs and specialized log data, causing overfitting. Furthermore, SFT's imbalanced loss computation often allows lengthy contexts to overwhelm critical, concise details in model answers, leading to hallucinations. To address these limitations, we propose R-Log, a novel reasoning-based paradigm that mirrors the structured, step-by-step analytical process of human engineers. This approach enhances generalizability by learning the underlying rules behind conclusions. We further employ Reinforcement Learning (RL) to optimize the model within a simulated O&M environment, thereby reducing hallucinations by directly rewarding correct outcomes. R-Log is first cold-started on a curated dataset of 2k+ reasoning trajectories, guided by 13 strategies from manual O&M practices, to establish an initial reasoning capability. This ability is then refined via RL using a joint reward function. Empirical evaluations on real-world logs show that R-Log outperforms existing methods across five log analysis tasks, particularly in unseen scenarios (by 228.05%). We also designed R-Log-fast with 5x speedup while keeping 93% of the efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03581v2">Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities. To our knowledge, this work is the first to explore training LLM agents for dynamic test-time compute allocation in sequential decision-making tasks, paving the way for more efficient, adaptive, and controllable agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25974v1">OpenID Connect for Agents (OIDC-A) 1.0: A Standard Extension for LLM-Based Agent Identity and Authorization</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 10 pages, 5 tables, 2 code listings. Specification proposal available at https://github.com/subramanya1997/oidc-a/
    </div>
    <details class="paper-abstract">
      OpenID Connect for Agents (OIDC-A) 1.0 is an extension to OpenID Connect Core 1.0 that provides a comprehensive framework for representing, authenticating, and authorizing LLM-based agents within the OAuth 2.0 ecosystem. As autonomous AI agents become increasingly prevalent in digital systems, there is a critical need for standardized protocols to establish agent identity, verify agent attestation, represent delegation chains, and enable fine-grained authorization based on agent attributes. This specification defines standard claims, endpoints, and protocols that address these requirements while maintaining compatibility with existing OAuth 2.0 and OpenID Connect infrastructure. The proposed framework introduces mechanisms for agent identity representation, delegation chain validation, attestation verification, and capability-based authorization, providing a foundation for secure and trustworthy agent-to-service interactions in modern distributed systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07680v2">Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue, but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MoE model, our method speeds up the training by 2.4$\times$ with competitive performance. Codes will be released at https://github.com/ModelTC/HBP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25919v1">Accelerating LLM Inference with Precomputed Query Storage</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference often suffers from high latency, particularly in resource-constrained environments such as on-device or edge deployments. To address this challenge, we present StorInfer, a novel storage-assisted LLM inference system that accelerates response time by precomputing and storing predictable query-response pairs offline. When a user query semantically matches a precomputed query, StorInfer bypasses expensive GPU inference and instantly returns the stored response, significantly reducing latency and compute costs. To maximize coverage and effectiveness, StorInfer employs an LLM-driven generator that adaptively produces diverse and deduplicated queries based on a given knowledge base. This is achieved via two techniques: adaptive query masking, which prevents regeneration of similar queries, and adaptive sampling, which dynamically tunes generation parameters to promote semantic diversity. The resulting query-response pairs are embedded and indexed using a disk-backed vector database to enable fast, similarity-based retrieval at runtime. Using this approach, we generated 150K unique precomputed pairs (taking up to 830 MB of storage space), achieving up to 17.3% latency reduction with no loss in response quality. Our evaluation across multiple QA datasets demonstrates the practicality and scalability of storage-assisted inference, especially in scenarios with predictable query distributions. StorInfer highlights a promising direction in leveraging storage as a primary enabler for efficient, low-latency LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12299v3">QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accept to ACLW 2025 (WOAH); fix typo
    </div>
    <details class="paper-abstract">
      The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25897v1">RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Humans often encounter role conflicts -- social dilemmas where the expectations of multiple roles clash and cannot be simultaneously fulfilled. As large language models (LLMs) become increasingly influential in human decision-making, understanding how they behave in complex social situations is essential. While previous research has evaluated LLMs' social abilities in contexts with predefined correct answers, role conflicts represent inherently ambiguous social dilemmas that require contextual sensitivity: the ability to recognize and appropriately weigh situational cues that can fundamentally alter decision priorities. To address this gap, we introduce RoleConflictBench, a novel benchmark designed to evaluate LLMs' contextual sensitivity in complex social dilemmas. Our benchmark employs a three-stage pipeline to generate over 13K realistic role conflict scenarios across 65 roles, systematically varying their associated expectations (i.e., their responsibilities and obligations) and situational urgency levels. By analyzing model choices across 10 different LLMs, we find that while LLMs show some capacity to respond to these contextual cues, this sensitivity is insufficient. Instead, their decisions are predominantly governed by a powerful, inherent bias related to social roles rather than situational information. Our analysis quantifies these biases, revealing a dominant preference for roles within the Family and Occupation domains, as well as a clear prioritization of male roles and Abrahamic religions across most evaluatee models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07448v2">Efficient Dynamic Ensembling for Multiple LLM Experts</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 IJCAI 2025
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated impressive performance across various language tasks. However, the strengths of LLMs can vary due to different architectures, model sizes, areas of training data, etc. Therefore, ensemble reasoning for the strengths of different LLM experts is critical to achieving consistent and satisfactory performance on diverse inputs across a wide range of tasks. However, existing LLM ensemble methods are either computationally intensive or incapable of leveraging complementary knowledge among LLM experts for various inputs. In this paper, we propose an efficient Dynamic Ensemble Reasoning paradigm, called DER to integrate the strengths of multiple LLM experts conditioned on dynamic inputs. Specifically, we model the LLM ensemble reasoning problem as a Markov Decision Process, wherein an agent sequentially takes inputs to request knowledge from an LLM candidate and passes the output to a subsequent LLM candidate. Moreover, we devise a reward function to train a DER-Agent to dynamically select an optimal answering route given the input questions, aiming to achieve the highest performance with as few computational resources as possible. Last, to fully transfer the expert knowledge from the prior LLMs, we develop a Knowledge Transfer Prompt that enables the subsequent LLM candidates to transfer complementary knowledge effectively. Experiments demonstrate that our method uses fewer computational resources to achieve better performance compared to state-of-the-art baselines. Code and appendix are available at https://github.com/Fhujinwu/DER
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17771v2">Speculating LLMs' Chinese Training Data Pollution from Their Tokens</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 23% long Chinese tokens (i.e., a token with more than two Chinese characters) are either porn or online gambling. We validate the accuracy of our speculation method on famous pre-training datasets like C4 and Pile. Then, considering GPT-4o, we speculate that the ratio of "Yui Hatano" related webpages in GPT-4o's training data is around 0.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25889v1">A Multimodal LLM Approach for Visual Question Answering on Multiparametric 3D Brain MRI</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 23 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We introduce mpLLM, a prompt-conditioned hierarchical mixture-of-experts (MoE) architecture for visual question answering over multi-parametric 3D brain MRI (mpMRI). mpLLM routes across modality-level and token-level projection experts to fuse multiple interrelated 3D modalities, enabling efficient training without image--report pretraining. To address limited image-text paired supervision, mpLLM integrates a synthetic visual question answering (VQA) protocol that generates medically relevant VQA from segmentation annotations, and we collaborate with medical experts for clinical validation. mpLLM outperforms strong medical VLM baselines by 5.3% on average across multiple mpMRI datasets. Our study features three main contributions: (1) the first clinically validated VQA dataset for 3D brain mpMRI, (2) a novel multimodal LLM that handles multiple interrelated 3D modalities, and (3) strong empirical results that demonstrate the medical utility of our methodology. Ablations highlight the importance of modality-level and token-level experts and prompt-conditioned routing. We have included our source code in the supplementary materials and will release our dataset upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15207v2">FlowRL: Matching Reward Distributions for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      We propose FlowRL: matching the full reward distribution via flow balancing instead of maximizing rewards in large language model (LLM) reinforcement learning (RL). Recent advanced reasoning models adopt reward-maximizing methods (\eg, PPO and GRPO), which tend to over-optimize dominant reward signals while neglecting less frequent but valid reasoning paths, thus reducing diversity. In contrast, we transform scalar rewards into a normalized target distribution using a learnable partition function, and then minimize the reverse KL divergence between the policy and the target distribution. We implement this idea as a flow-balanced optimization method that promotes diverse exploration and generalizable reasoning trajectories. We conduct experiments on math and code reasoning tasks: FlowRL achieves a significant average improvement of $10.0\%$ over GRPO and $5.1\%$ over PPO on math benchmarks, and performs consistently better on code reasoning tasks. These results highlight reward distribution-matching as a key step toward efficient exploration and diverse reasoning in LLM reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25885v1">SafeMind: Benchmarking and Mitigating Safety Risks in Embodied LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Embodied agents powered by large language models (LLMs) inherit advanced planning capabilities; however, their direct interaction with the physical world exposes them to safety vulnerabilities. In this work, we identify four key reasoning stages where hazards may arise: Task Understanding, Environment Perception, High-Level Plan Generation, and Low-Level Action Generation. We further formalize three orthogonal safety constraint types (Factual, Causal, and Temporal) to systematically characterize potential safety violations. Building on this risk model, we present SafeMindBench, a multimodal benchmark with 5,558 samples spanning four task categories (Instr-Risk, Env-Risk, Order-Fix, Req-Align) across high-risk scenarios such as sabotage, harm, privacy, and illegal behavior. Extensive experiments on SafeMindBench reveal that leading LLMs (e.g., GPT-4o) and widely used embodied agents remain susceptible to safety-critical failures. To address this challenge, we introduce SafeMindAgent, a modular Planner-Executor architecture integrated with three cascaded safety modules, which incorporate safety constraints into the reasoning process. Results show that SafeMindAgent significantly improves safety rate over strong baselines while maintaining comparable task completion. Together, SafeMindBench and SafeMindAgent provide both a rigorous evaluation suite and a practical solution that advance the systematic study and mitigation of safety risks in embodied LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25873v1">Lita: Light Agent Uncovers the Agentic Coding Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being applied to programming tasks, ranging from single-turn code completion to autonomous agents. Current code agent designs frequently depend on complex, hand-crafted workflows and tool sets. However, this reliance on elaborate scaffolding presents several challenges: agent performance becomes overly dependent on prompt tuning and custom design choices, heavy human intervention obscures a model's true underlying capabilities, and intricate pipelines are costly to build and maintain. Furthermore, optimizing complex task prompts increases the risk of data leakage. Currently, when introducing new models, LLM providers like OpenAI and Anthropic often publish benchmark scores to demonstrate their models' coding proficiency, but keep their proprietary evaluation frameworks confidential. To address these limitations, we introduce Lita (Lite Agent), which operationalizes liteness, a principle of minimizing manual design while retaining the essential elements of a fully autonomous agent. Lita enables a more faithful and unified evaluation without elaborate scaffolding. Experiments on the Aider Polyglot and SWE-Bench with frontier models demonstrate that Lita achieves competitive or superior performance compared to workflow-based and agentic baselines. Crucially, Lita also consumes fewer tokens and requires significantly less design effort. Our results suggest that Lita is sufficient to reveal the underlying coding competence of modern LLMs. Finally, we propose the Agent Complexity Law: the performance gap between agents of varying complexity, from simple to sophisticated designs, will shrink as the core model improves, ultimately converging to a negligible difference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13115v3">Dagger Behind Smile: Fool LLMs with a Happy Ending Story</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request. This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79% attack success rate on average. We also provide quantitative explanations for the success of HEA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25853v1">SAIL: SRAM-Accelerated LLM Inference System with Lookup-Table-based GEMV</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference requires substantial computational resources, yet CPU-based inference remains essential for democratizing AI due to the widespread availability of CPUs compared to specialized accelerators. However, efficient LLM inference on CPUs faces two fundamental challenges: (1) existing CPU architectures struggle with low-precision arithmetic required by quantized models, where optimal bit precision varies across models and layers; and (2) the memory-bound nature of the token generation phase creates severe performance bottlenecks. To address these challenges, we propose SAIL (SRAM-Accelerated Inference of LLMs), a CPU-based inference solution that efficiently supports arbitrary bit precisions with minimal overhead. SAIL integrates three key innovations: First, we introduce Batched LUT-based General Matrix-Vector Multiplication (LUT-GEMV) with SRAM-based processing-in-memory, enabling high data reuse through lookup tables and reducing memory movement. Second, our Pattern-Aware LUT optimization identifies and exploits redundancy in input activation patterns, reducing computation cycles by 13.8\%. Third, we develop an in-memory type conversion algorithm that leverages PIM's parallelism for efficient de-/quantization operations, alleviating pressure on CPU's vector units. Our architecture requires only 2\% hardware overhead and a single new instruction, while maintaining dual functionality as both compute and storage units. Experimental evaluations using a modified gem5 simulator demonstrate that SAIL achieves up to 10.7x speedup and 19.9x higher tokens per dollar compared to ARM Neoverse-N1 CPU baselines, and up to 7.04x better cost efficiency than NVIDIA V100 GPUs, establishing a practical path for efficient CPU-based LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25849v1">Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can self-improve through reinforcement learning, where they generate trajectories to explore and discover better solutions. However, this exploration process is computationally expensive, often forcing current methods to assign limited exploration budgets to each task. This uniform allocation creates problematic edge cases: easy tasks consistently succeed while difficult tasks consistently fail, both producing zero gradients during training updates for the widely used Group Relative Policy Optimization (GRPO). We address this problem from the lens of exploration budget allocation. Viewing each task's exploration as an "item" with a distinct "value" and "cost", we establish a connection to the classical knapsack problem. This formulation allows us to derive an optimal assignment rule that adaptively distributes resources based on the model's current learning status. When applied to GRPO, our method increases the effective ratio of non-zero policy gradients by 20-40% during training. Acting as a computational "free lunch", our approach could reallocate exploration budgets from tasks where learning is saturated to those where it is most impactful. This enables significantly larger budgets (e.g., 93 rollouts) for especially challenging problems, which would be computationally prohibitive under a uniform allocation. These improvements translate to meaningful gains on mathematical reasoning benchmarks, with average improvements of 2-4 points and peak gains of 9 points on specific tasks. Notably, achieving comparable performance with traditional homogeneous allocation would require about 2x the computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20462v4">TAMO: Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data in Cloud-Native Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Implementing large language models (LLMs)-driven root cause analysis (RCA) in cloud-native systems has become a key topic of modern software operations and maintenance. However, existing LLM-based approaches face three key challenges: multi-modality input constraint, context window limitation, and dynamic dependence graph. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data for fine-grained RCA, namely TAMO, including multimodality alignment tool, root cause localization tool, and fault types classification tool. In detail, TAMO unifies multi-modal observation data into time-aligned representations for cross-modal feature consistency. Based on the unified representations, TAMO then invokes its specialized root cause localization tool and fault types classification tool for further identifying root cause and fault type underlying system context. This approach overcomes the limitations of LLMs in processing real-time raw observational data and dynamic service dependencies, guiding the model to generate repair strategies that align with system context through structured prompt design. Experiments on two benchmark datasets demonstrate that TAMO outperforms state-of-the-art (SOTA) approaches with comparable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24836v2">Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data under-explored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training data. Rather than increasing data volume, our method re-optimizes existing samples to better align with the LLM's logical reasoning boundary. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25835v1">Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Test-time scaling enables large language models (LLMs) to improve performance on long-horizon reasoning tasks by allocating additional compute at inference. Tree-search-based approaches achieve state-of-the-art results in this setting, but they are notoriously inefficient, often an order of magnitude slower than simpler iterative methods. We introduce Chain-in-Tree (CiT), a plug-in framework that adaptively decides when to branch during search rather than branching at every step. CiT relies on lightweight Branching Necessity (BN) evaluation methods: BN-DP (Direct Prompting), where an auxiliary LLM directly judges whether a step requires branching, and BN-SC (Self-Consistency), which clusters multiple candidate actions to estimate agreement. We integrate CiT into three representative LLM-in-the-loop tree search frameworks: Tree of Thoughts (ToT-BS), ReST-MCTS, and RAP, and evaluate across GSM8K and Math500. Our results show that: (1) BN-DP consistently reduces token generation, model invocations, and runtime by 75-85 percent across all settings, with negligible accuracy loss and sometimes accuracy gains; (2) BN-SC typically yields substantial savings (up to 80 percent) but shows instability in 1-4 out of 14 settings, caused by a small subset of examples that produce very long reasoning steps; (3) the quality of auxiliary LLMs is critical, not only the BN evaluator in BN-DP, but also the models used in BN-SC for clustering and equivalence checking. When these roles are filled by smaller LLMs, performance degrades. Importantly, BN-SC does not require LLMs in domains with deterministic action spaces, where clustering can be done programmatically. We also provide a theoretical guarantee that BN-DP never increases LLM invocations relative to the baseline and release a unified implementation of CiT across ToT-BS, ReST-MCTS, and RAP to facilitate reproducibility and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23384v2">A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 System name updated and minor revisions
    </div>
    <details class="paper-abstract">
      LLM inference serving typically scales out with a two-tier architecture: a cluster router distributes requests to multiple inference engines, each of which then in turn performs its own internal scheduling. However, this commonly used paradigm suffers from critical, systemic inefficiency caused by the information gaps across two layers. At the cluster-layer, the router mainly relies on lagging, coarse-grained metrics, such as average latency and queue length to make decisions, resulting in "decision lag" that leads to suboptimal request routing. At the engine-layer, static heuristic scheduling policies cannot effectively handle the dynamic workloads, leading a poor balance between latency and throughput. Besides, these gaps may cause SLO violations and resource waste, especially in heterogeneous cloud environments. To bridge such gaps, we propose SynergySched, a cross-layer framework that shifts LLM serving system from reactive load balancing to predictive orchestration. The core of SynergySched lies in a structurally-informed online performance model that provides accurate, forward-looking per-step latency and capacity estimations. This model empowers two key components. At the engine-layer, LENS performs SLO-aware, adaptive scheduling, dynamically optimizing batching to meet SLOs under real-time loads. At the cluster-layer, PRISM uses predictive signals to perform state-driven routing, maximizing cluster-wide performance and SLO attainment. Performance evaluations show that SynergySched improves SLO attainment by 43% on average and achieves up to 3x throughput speedup in long-context and heterogeneous scenarios. Besides, we also deploy SynergySched on FlowGPT's clusters to demonstrate its advantages in production environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25818v1">VELA: An LLM-Hybrid-as-a-Judge Approach for Evaluating Long Image Captions</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      In this study, we focus on the automatic evaluation of long and detailed image captions generated by multimodal Large Language Models (MLLMs). Most existing automatic evaluation metrics for image captioning are primarily designed for short captions and are not suitable for evaluating long captions. Moreover, recent LLM-as-a-Judge approaches suffer from slow inference due to their reliance on autoregressive inference and early fusion of visual information. To address these limitations, we propose VELA, an automatic evaluation metric for long captions developed within a novel LLM-Hybrid-as-a-Judge framework. Furthermore, we propose LongCap-Arena, a benchmark specifically designed for evaluating metrics for long captions. This benchmark comprises 7,805 images, the corresponding human-provided long reference captions and long candidate captions, and 32,246 human judgments from three distinct perspectives: Descriptiveness, Relevance, and Fluency. We demonstrated that VELA outperformed existing metrics and achieved superhuman performance on LongCap-Arena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25795v1">Assessing Algorithmic Bias in Language-Based Depression Detection: A Comparison of DNN and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 7 pages, 1 figure. This paper has been accepted to the IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 2025), Georgia Institute of Technology, Atlanta, Georgia, October 26-29, 2025
    </div>
    <details class="paper-abstract">
      This paper investigates algorithmic bias in language-based models for automated depression detection, focusing on socio-demographic disparities related to gender and race/ethnicity. Models trained using deep neural networks (DNN) based embeddings are compared to few-shot learning approaches with large language models (LLMs), evaluating both performance and fairness on clinical interview transcripts from the Distress Analysis Interview Corpus/Wizard-of-Oz (DAIC-WOZ). To mitigate bias, fairness-aware loss functions are applied to DNN-based models, while in-context learning with varied prompt framing and shot counts is explored for LLMs. Results indicate that LLMs outperform DNN-based models in depression classification, particularly for underrepresented groups such as Hispanic participants. LLMs also exhibit reduced gender bias compared to DNN-based embeddings, though racial disparities persist. Among fairness-aware techniques for mitigating bias in DNN-based embeddings, the worst-group loss, which is designed to minimize loss for the worst-performing demographic group, achieves a better balance between performance and fairness. In contrast, the fairness-regularized loss minimizes loss across all groups but performs less effectively. In LLMs, guided prompting with ethical framing helps mitigate gender bias in the 1-shot setting. However, increasing the number of shots does not lead to further reductions in disparities. For race/ethnicity, neither prompting strategy nor increasing $N$ in $N$-shot learning effectively reduces disparities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13266v3">QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a central paradigm for training large language models (LLMs) in reasoning tasks. Yet recent studies question RL's ability to incentivize reasoning capacity beyond the base model. This raises a key challenge: how can RL be adapted to solve harder reasoning problems more effectively? To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 72.50% (+10.73%) on AIME24, 62.29% (+12.79%) on AIME25, and 41.67% (+10.11%) on HMMT25. Code, data and model are available at https://github.com/foreverlasting1202/QuestA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25773v1">V-HUB: A Visual-Centric Humor Understanding Benchmark for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      AI models capable of comprehending humor hold real-world promise -- for example, enhancing engagement in human-machine interactions. To gauge and diagnose the capacity of multimodal large language models (MLLMs) for humor understanding, we introduce v-HUB, a novel visual-centric video humor understanding benchmark. v-HUB comprises a curated collection of minimally verbal short videos, sourced from classic silent films and online resources, and reflecting real-world scenarios where humor can be appreciated purely through visual cues. Each video clip is paired with rich annotations, including captions, descriptions, and explanations, supporting evaluation tasks like caption matching and humor explanation. To broaden its applicability, we further construct an open-ended video QA task, making it readily integrable into existing video understanding benchmarks. We evaluate a diverse set of MLLMs, from specialized Video-LLMs to versatile OmniLLMs that can process audio, covering both open-source and proprietary domains. The experimental results expose the difficulties MLLMs face in comprehending humor from visual cues alone. For example, all models exhibit a marked performance drop on caption matching when moving from text-based to video-based evaluation (without audio). Our findings also demonstrate that incorporating audio helps with video humor understanding, highlighting the informativeness of sound and the promise of integrating richer modalities for complex video understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25760v1">TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated strong performance on factoid question answering, they are still prone to hallucination and untruthful responses, particularly when tasks demand information outside their parametric knowledge. Indeed, truthfulness requires more than accuracy -- models must also recognize uncertainty and abstain when unsure to avoid hallucinations. This presents a fundamental challenge for existing methods: approaches that optimize for accuracy often amplify hallucinations, while those that encourage abstention can become overly conservative, sacrificing correct answers. Both extremes ultimately compromise truthfulness. In this work, we present TruthRL, a general reinforcement learning (RL) framework that directly optimizes the truthfulness of LLMs. Specifically, we implement TruthRL using GRPO with a simple yet effective ternary reward that distinguishes correct answers, hallucinations, and abstentions. It incentivizes models to reduce hallucinations not only by providing correct responses, but also by enabling abstention when uncertain, thereby improving truthfulness. Extensive experiments across four knowledge-intensive benchmarks show that, compared to vanilla RL, TruthRL significantly reduces hallucinations by 28.9% and improves truthfulness by 21.1%, with consistent gains across various backbone models (e.g., Qwen, Llama) under both retrieval and non-retrieval setups. In-depth ablation study demonstrates that vanilla accuracy-driven methods, such as supervised fine-tuning or RL with a binary reward, struggle to balance factual correctness and uncertainty. In contrast, our proposed truthfulness-driven TruthRL achieves strong performance in both accuracy and truthfulness, underscoring the importance of learning objective design for developing truthful LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25743v1">Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data. However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24765v2">From Ambiguity to Verdict: A Semiotic-Grounded Multi-Perspective Agent for LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Logical reasoning is a fundamental capability of large language models (LLMs). However, existing studies largely overlook the interplay between logical complexity and semantic complexity, resulting in methods that struggle to address challenging scenarios involving abstract propositions, ambiguous contexts, and conflicting stances, which are central to human reasoning. For this gap, we propose LogicAgent, a semiotic-square-guided framework designed to jointly address logical complexity and semantic complexity. LogicAgent explicitly performs multi-perspective deduction in first-order logic (FOL), while mitigating vacuous reasoning through existential import checks that incorporate a three-valued decision scheme (True, False, Uncertain) to handle boundary cases more faithfully. Furthermore, to overcome the semantic simplicity and low logical complexity of existing datasets, we introduce RepublicQA, a benchmark that reaches college-level difficulty (FKGL = 11.94) and exhibits substantially greater lexical and structural diversity than prior benchmarks. RepublicQA is grounded in philosophical concepts, featuring abstract propositions and systematically organized contrary and contradictory relations, making it the most semantically rich resource for evaluating logical reasoning. Experiments demonstrate that LogicAgent achieves state-of-the-art performance on RepublicQA, with a 6.25% average gain over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional 7.05% average gain. These results highlight the strong effectiveness of our semiotic-grounded multi-perspective reasoning in boosting LLMs' logical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25725v1">Atomic Thinking of LLMs: Decoupling and Exploring Mathematical Reasoning Abilities</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated outstanding performance in mathematical reasoning capabilities. However, we argue that current large-scale reasoning models primarily rely on scaling up training datasets with diverse mathematical problems and long thinking chains, which raises questions about whether LLMs genuinely acquire mathematical concepts and reasoning principles or merely remember the training data. In contrast, humans tend to break down complex problems into multiple fundamental atomic capabilities. Inspired by this, we propose a new paradigm for evaluating mathematical atomic capabilities. Our work categorizes atomic abilities into two dimensions: (1) field-specific abilities across four major mathematical fields, algebra, geometry, analysis, and topology, and (2) logical abilities at different levels, including conceptual understanding, forward multi-step reasoning with formal math language, and counterexample-driven backward reasoning. We propose corresponding training and evaluation datasets for each atomic capability unit, and conduct extensive experiments about how different atomic capabilities influence others, to explore the strategies to elicit the required specific atomic capability. Evaluation and experimental results on advanced models show many interesting discoveries and inspirations about the different performances of models on various atomic capabilities and the interactions between atomic capabilities. Our findings highlight the importance of decoupling mathematical intelligence into atomic components, providing new insights into model cognition and guiding the development of training strategies toward a more efficient, transferable, and cognitively grounded paradigm of "atomic thinking".
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00083v2">A Survey on Code Generation with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Work in progress (V2)
    </div>
    <details class="paper-abstract">
      Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15463v4">Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 EMNLP 2025 Main Paper
    </div>
    <details class="paper-abstract">
      Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24116v2">Dual-Scale World Models for LLM Agents Towards Hard-Exploration Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      LLM-based agents have seen promising advances, yet they are still limited in "hard-exploration" tasks requiring learning new knowledge through exploration. We present GLoW, a novel approach leveraging dual-scale world models, maintaining a trajectory frontier of high-value discoveries at the global scale, while learning from local trial-and-error in exploration through a Multi-path Advantage Reflection mechanism which infers advantage-based progress signals to guide exploration. To evaluate our framework for hard-exploration, we tackle the Jericho benchmark suite of text-based games, where GLoW achieves a new state-of-theart performance for LLM-based approaches. Compared to state-of-the-art RLbased methods, our approach achieves comparable performance while requiring 100-800x fewer environment interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25694v1">HNote: Extending YNote with Hexadecimal Encoding for Fine-Tuning LLMs in Music Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have created new opportunities for symbolic music generation. However, existing formats such as MIDI, ABC, and MusicXML are either overly complex or structurally inconsistent, limiting their suitability for token-based learning architectures. To address these challenges, we propose HNote, a novel hexadecimal-based notation system extended from YNote, which encodes both pitch and duration within a fixed 32-unit measure framework. This design ensures alignment, reduces ambiguity, and is directly compatible with LLM architectures. We converted 12,300 Jiangnan-style songs generated from traditional folk pieces from YNote into HNote, and fine-tuned LLaMA-3.1(8B) using parameter-efficient LoRA. Experimental results show that HNote achieves a syntactic correctness rate of 82.5%, and BLEU and ROUGE evaluations demonstrate strong symbolic and structural similarity, producing stylistically coherent compositions. This study establishes HNote as an effective framework for integrating LLMs with cultural music modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01245v4">Towards Agentic OS: An LLM Agent Framework for Linux Schedulers</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"), thereby separating the optimization problem into two stages: goal-inference and policy-synthesis. Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis. We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in https://github.com/eunomia-bpf/schedcp
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02329v3">ReSpark: Leveraging Previous Data Reports as References to Generate New Reports with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Creating data reports is a labor-intensive task involving iterative data exploration, insight extraction, and narrative construction. A key challenge lies in composing the analysis logic-from defining objectives and transforming data to identifying and communicating insights. Manually crafting this logic can be cognitively demanding. While experienced analysts often reuse scripts from past projects, finding a perfect match for a new dataset is rare. Even when similar analyses are available online, they usually share only results or visualizations, not the underlying code, making reuse difficult. To address this, we present ReSpark, a system that leverages large language models (LLMs) to reverse-engineer analysis logic from existing reports and adapt it to new datasets. By generating draft analysis steps, ReSpark provides a warm start for users. It also supports interactive refinement, allowing users to inspect intermediate outputs, insert objectives, and revise content. We evaluate ReSpark through comparative and user studies, demonstrating its effectiveness in lowering the barrier to generating data reports without relying on existing analysis code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25676v1">Explainable Fault Localization for Programming Assignments via LLM-Guided Annotation</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted by ASE'2025
    </div>
    <details class="paper-abstract">
      Providing timely and personalized guidance for students' programming assignments, offers significant practical value for helping students complete assignments and enhance their learning. In recent years, various automated Fault Localization (FL) techniques have demonstrated promising results in identifying errors in programs. However, existing FL techniques face challenges when applied to educational contexts. Most approaches operate at the method level without explanatory feedback, resulting in granularity too coarse for students who need actionable insights to identify and fix their errors. While some approaches attempt line-level fault localization, they often depend on predicting line numbers directly in numerical form, which is ill-suited to LLMs. To address these challenges, we propose FLAME, a fine-grained, explainable Fault Localization method tailored for programming assignments via LLM-guided Annotation and Model Ensemble. FLAME leverages rich contextual information specific to programming assignments to guide LLMs in identifying faulty code lines. Instead of directly predicting line numbers, we prompt the LLM to annotate faulty code lines with detailed explanations, enhancing both localization accuracy and educational value. To further improve reliability, we introduce a weighted multi-model voting strategy that aggregates results from multiple LLMs to determine the suspiciousness of each code line. Extensive experimental results demonstrate that FLAME outperforms state-of-the-art fault localization baselines on programming assignments, successfully localizing 207 more faults at top-1 over the best-performing baseline. Beyond educational contexts, FLAME also generalizes effectively to general-purpose software codebases, outperforming all baselines on the Defects4J benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25666v1">Nudging the Boundaries of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Code release in preparation
    </div>
    <details class="paper-abstract">
      Current online reinforcement learning (RL) algorithms like GRPO share a key limitation in LLM reasoning: they cannot learn from problems that are "unsolvable" to the model. In other words, they can only improve performance on problems where the model is capable of exploring the correct answer. Consequently, the model's "upper limit" remains unchanged after RL training, even though the likelihood of solving easier, solvable problems may increase. These hard samples cannot contribute to training, as no rollouts yield rewards and thus no gradients are produced. To unlock learning from these hard samples, we propose NuRL, a "nudging" method that aims to push the upper bound of LLM reasoning using self-generated hints, i.e., abstract cues that help reduce the problem difficulty for the model. Given a question and its gold answer, the model generates a CoT and then produces a hint containing the core knowledge needed to solve the problem. During training, we generate G rollouts from the base policy and use the pass rate to decide whether the hint should be injected. For hard samples with a 0% pass rate, we inject the hint and regenerate a new batch of trajectories. This yields two benefits: (1) the hint boosts pass rates (from 0% to non-zero), thereby introducing training signals for previously unsolvable samples, and (2) the hints are self-generated, avoiding distributional shift and do not rely on external models. NuRL achieves consistent improvements across 6 benchmarks and 3 models, while remaining complementary to test-time scaling. Notably, NuRL can raise the model's upper limit, whereas GRPO leaves pass@1024 unchanged from the base model. Furthermore, we present a systematic study of what makes an effective hint and when hints are most useful. Interestingly, the best hints are abstract and high-level, and are most beneficial when applied necessarily and after GRPO has converged.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12686v2">Understanding Formal Reasoning Failures in LLMs as Abstract Interpreters</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for program verification, and yet little is known about \emph{how} they reason about program semantics during this process. In this work, we focus on abstract interpretation based-reasoning for invariant generation and introduce two novel prompting strategies that aim to elicit such reasoning from LLMs. We evaluate these strategies across several state-of-the-art LLMs on 22 programs from the SV-COMP benchmark suite widely used in software verification. We analyze both the soundness of the generated invariants and the key thematic patterns in the models' reasoning errors. This work aims to highlight new research opportunities at the intersection of LLMs and program verification for applying LLMs to verification tasks and advancing their reasoning capabilities in this application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23143v2">MathBode: Frequency-Domain Fingerprints of LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      This paper presents MathBode, a dynamic diagnostic for mathematical reasoning in large language models (LLMs). Instead of one-shot accuracy, MathBode treats each parametric problem as a system: we drive a single parameter sinusoidally and fit first-harmonic responses of model outputs and exact solutions. This yields interpretable, frequency-resolved metrics -- gain (amplitude tracking) and phase (lag) -- that form Bode-style fingerprints. Across five closed-form families (linear solve, ratio/saturation, compound interest, 2x2 linear systems, similar triangles), the diagnostic surfaces systematic low-pass behavior and growing phase lag that accuracy alone obscures. We compare several models against a symbolic baseline that calibrates the instrument ($G \approx 1$, $\phi \approx 0$). Results separate frontier from mid-tier models on dynamics, providing a compact, reproducible protocol that complements standard benchmarks with actionable measurements of reasoning fidelity and consistency. We open-source the dataset and code to enable further research and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25626v1">LLM-Powered Code Analysis and Optimization for Gaussian Splatting Kernels</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is a transformative technique with profound implications on novel view synthesis and real-time rendering. Given its importance, there have been many attempts to improve its performance. However, with the increasing complexity of GPU architectures and the vast search space of performance-tuning parameters, it is a challenging task. Although manual optimizations have achieved remarkable speedups, they require domain expertise and the optimization process can be highly time consuming and error prone. In this paper, we propose to exploit large language models (LLMs) to analyze and optimize Gaussian splatting kernels. To our knowledge, this is the first work to use LLMs to optimize highly specialized real-world GPU kernels. We reveal the intricacies of using LLMs for code optimization and analyze the code optimization techniques from the LLMs. We also propose ways to collaborate with LLMs to further leverage their capabilities. For the original 3DGS code on the MipNeRF360 datasets, LLMs achieve significant speedups, 19% with Deepseek and 24% with GPT-5, demonstrating the different capabilities of different LLMs. By feeding additional information from performance profilers, the performance improvement from LLM-optimized code is enhanced to up to 42% and 38% on average. In comparison, our best-effort manually optimized version can achieve a performance improvement up to 48% and 39% on average, showing that there are still optimizations beyond the capabilities of current LLMs. On the other hand, even upon a newly proposed 3DGS framework with algorithmic optimizations, Seele, LLMs can still further enhance its performance by 6%, showing that there are optimization opportunities missed by domain experts. This highlights the potential of collaboration between domain experts and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25624v1">STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      As LLMs advance into autonomous agents with tool-use capabilities, they introduce security challenges that extend beyond traditional content-based LLM safety concerns. This paper introduces Sequential Tool Attack Chaining (STAC), a novel multi-turn attack framework that exploits agent tool use. STAC chains together tool calls that each appear harmless in isolation but, when combined, collectively enable harmful operations that only become apparent at the final execution step. We apply our framework to automatically generate and systematically evaluate 483 STAC cases, featuring 1,352 sets of user-agent-environment interactions and spanning diverse domains, tasks, agent types, and 10 failure modes. Our evaluations show that state-of-the-art LLM agents, including GPT-4.1, are highly vulnerable to STAC, with attack success rates (ASR) exceeding 90% in most cases. The core design of STAC's automated framework is a closed-loop pipeline that synthesizes executable multi-step tool chains, validates them through in-environment execution, and reverse-engineers stealthy multi-turn prompts that reliably induce agents to execute the verified malicious sequence. We further perform defense analysis against STAC and find that existing prompt-based defenses provide limited protection. To address this gap, we propose a new reasoning-driven defense prompt that achieves far stronger protection, cutting ASR by up to 28.8%. These results highlight a crucial gap: defending tool-enabled agents requires reasoning over entire action sequences and their cumulative effects, rather than evaluating isolated prompts or responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02296v2">Memorize or Generalize? Evaluating LLM Code Generation with Code Rewriting</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated exceptional code generation capabilities. However, there is a growing debate whether LLMs are mostly doing memorization (i.e., replicating or reusing large parts of their training data) versus generalization (i.e., beyond training data). Existing evaluations largely proxy memorization with surface/structural similarity, thereby conflating benign reuse of repeated code with harmful recall and neglecting task correctness under semantic variation. We define harmful memorization behaviorally as failure at high similarity and introduce a semantic perturbation code rewriting, which rewrites a semantically different answer at a similar difficulty level for a given coding task, then reverse-engineers a novel coding task. We further propose Memorization Risk Index (MRI), a normalized score that combines two signals: (i) how similar the model's answer for the rewritten task is to the original ground-truth solution, and (ii) how much performance drops from the original task to its rewritten counterpart. MRI is high only when both conditions hold -- when the model outputs similar code but fails the perturbed task -- thereby capturing harmful memorization rather than benign reuse of repeated code. Empirical evaluations on code generation benchmarks MBPP+ and BigCodeBench reveal that (1) memorization does not increase with larger models and in many cases alleviates as they scale; (2) supervised fine-tuning (SFT) improves accuracy while introduces memorization; (3) reinforcement learning with proximal policy optimization (PPO) achieves a more balanced trade-off between memorization and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00245v2">The Rarity Blind Spot: A Framework for Evaluating Statistical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Effective decision-making often relies on identifying what makes each candidate distinctive. While existing benchmarks for LLMs emphasize retrieving or summarizing information relevant to a given query, they do not evaluate a model's ability to identify globally distinctive features across a set of documents. We introduce Distinctive Feature Mining (DFM), a new task that challenges models to analyze a small-to-medium collection (10-40 documents) and surface features that are rare in the global context (e.g., appearing in less than 10% of documents). This setting mirrors real-world scenarios such as candidate selection or product differentiation, where statistical reasoning, not retrieval, is key. To enable systematic evaluation of this capability, we present DiFBench, a configurable benchmark creation framework with controllable parameters such as document set size and distinctiveness thresholds. Using DiFBench, we perform a large-scale assessment of distinctive feature mining across ten state-of-the-art LLMs. Our findings reveal a significant performance gap between general-purpose and reasoning-enhanced models. All models, however, substantially degrade as the task complexity and document count increase. We also find that a common failure mode is misidentifying frequent features as distinctive. These insights reveal core limitations in contemporary LLMs' abilities to perform fine-grained, statistical reasoning and rarity detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23234v2">$p$-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10197v2">MathConstruct: Challenging LLM Reasoning with Constructive Proofs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate impressive performance in mathematics, existing math benchmarks come with significant limitations. Many focus on problems with fixed ground-truth answers, and are often saturated due to problem simplicity or the viability of guessing or memorization. Crucially, they capture only a narrow subset of relevant math problems. To address this research gap, we introduce MathConstruct, a new benchmark of 121 challenging problems sourced from various math competitions, which targets constructive proofs, a widely encountered problem type requiring the construction of mathematical objects with specific properties. These proofs are particularly suitable for LLM evaluation, as solution correctness can be easily verified. Our automated verifiers also enable MathConstruct to generate problem variations, used to evaluate robustness. State-of-the-art LLMs solve only 60% of MathConstruct problems, highlighting its complexity and importance for LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14435v2">Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted for EMNLP Conference
    </div>
    <details class="paper-abstract">
      As organizations increasingly rely on AI systems for decision support in sustainability contexts, it becomes critical to understand the inherent biases and perspectives embedded in Large Language Models (LLMs). This study systematically investigates how five state-of-the-art LLMs -- Claude, DeepSeek, GPT, LLaMA, and Mistral - conceptualize sustainability and its relationship with AI. We administered validated, psychometric sustainability-related questionnaires - each 100 times per model -- to capture response patterns and variability. Our findings revealed significant inter-model differences: For example, GPT exhibited skepticism about the compatibility of AI and sustainability, whereas LLaMA demonstrated extreme techno-optimism with perfect scores for several Sustainable Development Goals (SDGs). Models also diverged in attributing institutional responsibility for AI and sustainability integration, a results that holds implications for technology governance approaches. Our results demonstrate that model selection could substantially influence organizational sustainability strategies, highlighting the need for awareness of model-specific biases when deploying LLMs for sustainability-related decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22223v3">Secure coding for web applications: Frameworks, challenges, and the role of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 11 pages, 5 figures, 3 tables, 6 listings
    </div>
    <details class="paper-abstract">
      Secure coding is a critical yet often overlooked practice in software development. Despite extensive awareness efforts, real-world adoption remains inconsistent due to organizational, educational, and technical barriers. This paper provides a comprehensive review of secure coding practices across major frameworks and domains, including web development, DevSecOps, and cloud security. It introduces a structured framework comparison and categorizes threats aligned with the OWASP Top 10. Additionally, we explore the rising role of Large Language Models (LLMs) in evaluating and recommending secure code, presenting a reproducible case study across four major vulnerability types. This paper offers practical insights for researchers, developers, and educators on integrating secure coding into real-world development processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01750v2">Communication-Aware Knowledge Distillation for Federated LLM Fine-Tuning over Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 accepted by Globecom 2025
    </div>
    <details class="paper-abstract">
      Federated learning (FL) for large language models (LLMs) offers a privacy-preserving scheme, enabling clients to collaboratively fine-tune locally deployed LLMs or smaller language models (SLMs) without exchanging raw data. While parameter-sharing methods in traditional FL models solves number of technical challenges, they still incur high communication overhead and struggle with adapting to heterogeneous model architectures. Federated distillation, a framework for mutual knowledge transfer via shared logits, typically offers lower communication overhead than parameter-sharing methods. However, transmitting logits from LLMs remains challenging for bandwidth-limited clients due to their high dimensionality. In this work, we focus on a federated LLM distillation with efficient communication overhead. To achieve this, we first propose an adaptive Top-k logit selection mechanism, dynamically sparsifying logits according to real-time communication conditions. Then to tackle the dimensional inconsistency introduced by the adaptive sparsification, we design an adaptive logits aggregation scheme, effectively alleviating the artificial and uninformative inputs introduced by conventional zero-padding methods. Finally, to enhance the distillation effect, we incorporate LoRA-adapted hidden-layer projection from LLM into the distillation loss, reducing the communication overhead further while providing richer representation. Experimental results demonstrate that our scheme achieves superior performance compared to baseline methods while effectively reducing communication overhead by approximately 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23808v2">Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      A prevailing view in Reinforcement Learning for Verifiable Rewards (RLVR) interprets recent progress through the lens of an exploration-exploitation trade-off, a perspective largely shaped by token-level metrics. We re-examine this perspective, proposing that this perceived trade-off may not be a fundamental constraint but rather an artifact of the measurement level. To investigate this, we shift the analysis to the semantically rich hidden-state space, adopting Effective Rank (ER) to quantify exploration and proposing its novel first- and second-order derivatives, named Effective Rank Velocity (ERV) and Effective Rank Acceleration (ERA), to capture exploitation dynamics. Our analysis reveals that at the hidden-state level, exploration and exploitation could be decoupled (Sec. 4). This finding reveals an opportunity to enhance both capacities simultaneously. This insight motivates our method, Velocity-Exploiting Rank-Learning (VERL), the first to operationalize the principle of synergistic exploration-exploitation enhancement by directly shaping the RL advantage function. The key innovation is leveraging the theoretically stable ERA as a predictive meta-controller to create a synergistic, dual-channel incentive structure. Instead of forcing a trade-off, VERL prospectively amplifies rewards for exploration to preempt overconfidence and reinforces exploitative gains to consolidate reasoning. Experiments across diverse LLMs and reasoning benchmarks show consistent gains, including up to 21.4% absolute accuracy improvement on the challenging Gaokao 2024 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01285v1">LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in data science, yet their practical deployment is often constrained by the challenge of discovering relevant data within large heterogeneous data lakes. Existing methods struggle with this: single-agent systems are quickly overwhelmed by large, heterogeneous files in the large data lakes, while multi-agent systems designed based on a master-slave paradigm depend on a rigid central controller for task allocation that requires precise knowledge of each sub-agent's capabilities. To address these limitations, we propose a novel multi-agent communication paradigm inspired by the blackboard architecture for traditional AI models. In this framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents -- either responsible for a partition of the data lake or general information retrieval -- volunteer to respond based on their capabilities. This design improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents' expertise. We evaluate our method on three benchmarks that require explicit data discovery: KramaBench and modified versions of DS-Bench and DA-Code to incorporate data discovery. Experimental results demonstrate that the blackboard architecture substantially outperforms baselines, including RAG and the master-slave multi-agent paradigm, achieving between 13% to 57% relative improvement in end-to-end task success and up to a 9% relative gain in F1 score for data discovery over the best-performing baselines across both proprietary and open-source LLMs. Our findings establish the blackboard paradigm as a scalable and generalizable communication framework for multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00324v1">Which Programming Language and Model Work Best With LLM-as-a-Judge For Code Retrieval?</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted as a full paper at SIGIR-AP 2025
    </div>
    <details class="paper-abstract">
      Code search is an important information retrieval application. Benefits of better code search include faster new developer on-boarding, reduced software maintenance, and ease of understanding for large repositories. Despite improvements in search algorithms and search benchmarks, the domain of code search has lagged behind. One reason is the high cost of human annotation for code queries and answers. While humans may annotate search results in general text QA systems, code annotations require specialized knowledge of a programming language (PL), as well as domain specific software engineering knowledge. In this work we study the use of Large Language Models (LLMs) to retrieve code at the level of functions and to generate annotations for code search results. We compare the impact of the retriever representation (sparse vs. semantic), programming language, and LLM by comparing human annotations across several popular languages (C, Java, Javascript, Go, and Python). We focus on repositories that implement common data structures likely to be implemented in any PLs. For the same human annotations, we compare several LLM-as-a-Judge models to evaluate programming language and other affinities between LLMs. We find that the chosen retriever and PL exhibit affinities that can be leveraged to improve alignment of human and AI relevance determinations, with significant performance implications. We also find differences in representation (sparse vs. semantic) across PLs that impact alignment of human and AI relevance determinations. We propose using transpilers to bootstrap scalable code search benchmark datasets in other PLs and in a case study demonstrate that human-AI relevance agreement rates largely match the (worst case) human-human agreement under study. The application code used in this work is available at \href{https://github.com/rlucas7/code-searcher/}{this github repo}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00311v1">CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Security Operations Centers (SOCs) are overwhelmed by tens of thousands of daily alerts, with only a small fraction corresponding to genuine attacks. This overload creates alert fatigue, leading to overlooked threats and analyst burnout. Classical detection pipelines are brittle and context-poor, while recent LLM-based approaches typically rely on a single model to interpret logs, retrieve context, and adjudicate alerts end-to-end -- an approach that struggles with noisy enterprise data and offers limited transparency. We propose CORTEX, a multi-agent LLM architecture for high-stakes alert triage in which specialized agents collaborate over real evidence: a behavior-analysis agent inspects activity sequences, evidence-gathering agents query external systems, and a reasoning agent synthesizes findings into an auditable decision. To support training and evaluation, we release a dataset of fine-grained SOC investigations from production environments, capturing step-by-step analyst actions and linked tool outputs. Across diverse enterprise scenarios, CORTEX substantially reduces false positives and improves investigation quality over state-of-the-art single-agent LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00276v1">SafePassage: High-Fidelity Information Extraction with Black Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Black box large language models (LLMs) make information extraction (IE) easy to configure, but hard to trust. Unlike traditional information extraction pipelines, the information "extracted" is not guaranteed to be grounded in the document. To prevent this, this paper introduces the notion of a "safe passage": context generated by the LLM that is both grounded in the document and consistent with the extracted information. This is operationalized via a three-step pipeline, SafePassage, which consists of: (1) an LLM extractor that generates structured entities and their contexts from a document, (2) a string-based global aligner, and (3) a scoring model. Results show that using these three parts in conjunction reduces hallucinations by up to 85% on information extraction tasks with minimal risk of flagging non-hallucinations. High agreement between the SafePassage pipeline and human judgments of extraction quality mean that the pipeline can be dually used to evaluate LLMs. Surprisingly, results also show that using a transformer encoder fine-tuned on a small number of task-specific examples can outperform an LLM scoring model at flagging unsafe passages. These annotations can be collected in as little as 1-2 hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00268v1">Efficient Layer-wise LLM Fine-tuning for Revision Intention Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 In The Conference on Empirical Methods in Natural Language Processing (EMNLP), November 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown extraordinary success across various text generation tasks; however, their potential for simple yet essential text classification remains underexplored, as LLM pre-training tends to emphasize generation over classification. While LLMs with instruction tuning can transform classification into a generation task, they often struggle to categorize nuanced texts. One such example is text revision, which involves nuanced edits between pairs of texts. Although simply fine-tuning LLMs for revision classification seems plausible, it requires a large amount of revision annotations, which are exceptionally expensive and scarce in the community. To address this issue, we introduce a plug-and-play layer-wise parameter-efficient fine-tuning (PEFT) framework, i.e., IR-Tuning, which fine-tunes a subset of important LLM layers that are dynamically selected based on their gradient norm distribution, while freezing those of redundant layers. Extensive experiments suggest that IR-Tuning surpasses several layer-wise PEFT baselines over diverse text revisions, while achieving fast convergence, low GPU memory consumption, and effectiveness on small revision corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00206v1">LoRAFusion: Efficient LoRA Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 Accepted by EuroSys 2026
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has become the leading Parameter-Efficient Fine-Tuning (PEFT) method for Large Language Models (LLMs), as it significantly reduces GPU memory usage while maintaining competitive fine-tuned model quality on downstream tasks. Despite these benefits, we identify two key inefficiencies in existing LoRA fine-tuning systems. First, they incur substantial runtime overhead due to redundant memory accesses on large activation tensors. Second, they miss the opportunity to concurrently fine-tune multiple independent LoRA adapters that share the same base model on the same set of GPUs. This leads to missed performance gains such as reduced pipeline bubbles, better communication overlap, and improved GPU load balance. To address these issues, we introduce LoRAFusion, an efficient LoRA fine-tuning system for LLMs. At the kernel level, we propose a graph-splitting method that fuses memory-bound operations. This design eliminates unnecessary memory accesses and preserves the performance of compute-bound GEMMs without incurring the cost of recomputation or synchronization. At the scheduling level, LoRAFusion introduces an adaptive batching algorithm for multi-job fine-tuning. It first splits LoRA adapters into groups to intentionally stagger batch execution across jobs, and then solves a bin-packing problem within each group to generate balanced, dependency-aware microbatches. LoRAFusion achieves up to $1.96\times$ ($1.47\times$ on average) end-to-end speedup compared to Megatron-LM, and up to $1.46\times$ ($1.29\times$ on average) improvement over mLoRA, the state-of-the-art multi-LoRA fine-tuning system. Our fused kernel achieves up to $1.39\times$ ($1.27\times$ on average) kernel performance improvement and can directly serve as a plug-and-play replacement in existing LoRA systems. We open-source LoRAFusion at https://github.com/CentML/lorafusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00202v1">RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Today's LLM ecosystem comprises a wide spectrum of models that differ in size, capability, and cost. No single model is optimal for all scenarios; hence, LLM routers have become essential for selecting the most appropriate model under varying circumstances. However, the rapid emergence of various routers makes choosing the right one increasingly challenging. To address this problem, we need a comprehensive router comparison and a standardized leaderboard, similar to those available for models. In this work, we introduce RouterArena, the first open platform enabling comprehensive comparison of LLM routers. RouterArena has (1) a principally constructed dataset with broad knowledge domain coverage, (2) distinguishable difficulty levels for each domain, (3) an extensive list of evaluation metrics, and (4) an automated framework for leaderboard updates. Leveraging our framework, we have produced the initial leaderboard with detailed metrics comparison as shown in Figure 1. We will make our platform open to the public soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00194v1">GRPO-$λ$: Credit Assignment improves LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed for tasks requiring complex reasoning, prompting significant interest in improving their reasoning abilities through post-training. Especially RL based methods using verifiable reward, like the state-of-the-art GRPO, have shown to tremendously improve reasoning behaviors when applied as post-training methods. However, the lack of an explicit reward or critic model limits GRPO's ability to assign fine-grained credit across token sequences. In this work, we present GRPO-$\lambda$, a novel extension to GRPO that enhances credit assignment in RL finetuning of LLMs for complex reasoning tasks. We approximate learning from $\lambda$-return with a reformulation of eligibility traces using token-level log-probabilities applied after each sequence generation, and a novel critic-free approximation of the temporal-difference error. We introduce a few variations for the weighting of the $\lambda$-return, and their applications to the eligibility-trace, where all the variations provide significant gains over GRPO. We compare GRPO-$\lambda$ against GRPO by training models from 1.5B to 7B parameters on $4$ different math reasoning datasets. The training plots demonstrate 30-40% improved performance during RL training on both LLaMA-3.1 and Qwen-2.5 architectures. Finally, we show that with GRPO-$\lambda$, the resulting average performance on AIME24, Math500, OlympiadMath, MinervaMath, and AMC improves over GRPO by over $3$ points and a $4.5$ points improvement on the 7B model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00177v1">Personalized Reasoning: Just-In-Time Personalization and Why LLMs Fail At It</a></div>
    <div class="paper-meta">
      📅 2025-09-30
      | 💬 57 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Current large language model (LLM) development treats task-solving and preference alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user's needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to identify what they don't know about user preferences, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly -- a complicated chain of cognitive processes which we term personalized reasoning. We introduce PREFDISCO, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse preferences. Our framework creates scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs effectively. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PREFDISCO establishes personalized reasoning as a measurable research frontier and reveals fundamental limitations in current LLMs' interactive capabilities, providing a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01276v1">LLM Based Sentiment Classification From Bangladesh E-Commerce Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      Sentiment analysis is an essential part of text analysis, which is a larger field that includes determining and evaluating the author's emotional state. This method is essential since it makes it easier to comprehend consumers' feelings, viewpoints, and preferences holistically. The introduction of large language models (LLMs), such as Llama, has greatly increased the availability of cutting-edge model applications, such as sentiment analysis. However, accurate sentiment analysis is hampered by the intricacy of written language and the diversity of languages used in evaluations. The viability of using transformer-based BERT models and other LLMs for sentiment analysis from Bangladesh e commerce reviews is investigated in this paper. A subset of 4000 samples from the original dataset of Bangla and English customer reviews was utilized to fine-tune the model. The fine tuned Llama-3.1-8B model outperformed other fine-tuned models, including Phi-3.5-mini-instruct, Mistral-7B-v0.1, DistilBERT-multilingual, mBERT, and XLM-R-base, with an overall accuracy, precision, recall, and F1 score of 95.5%, 93%, 88%, 90%. The study emphasizes how parameter efficient fine-tuning methods (LoRA and PEFT) can lower computational overhead and make it appropriate for contexts with limited resources. The results show how LLMs can
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2507.10540v2">FusionFactory: Fusing LLM Capabilities with Multi-LLM Log Data</a></div>
    <div class="paper-meta">
      📅 2025-09-30
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a diverse landscape of models, each excelling at different tasks. This diversity drives researchers to employ multiple LLMs in practice, leaving behind valuable multi-LLM log data. This naturally leads to the question of whether such logs can be fully leveraged to fuse LLMs' complementary capabilities. Although prior work has explored various strategies for integrating multiple LLMs, we argue that practical fusion must meet two essential requirements: (1) compatibility with real-world serving scenarios (e.g., local and API-based serving), and (2) flexibility to operate at different stages of the LLM pipeline to meet varied user needs (e.g., fine-tuning and inference stages). To this end, we introduce LLMFusionBench, a large-scale benchmark for LLM fusion that spans 14 tasks across five domains, with responses from 20 open-source LLMs (8B--671B) totaling 103M tokens. Building on LLMFusionBench, we propose FusionFactory, a systematic framework with three elaborated levels: (1) query-level fusion via tailored LLM routers, (2) thought-level fusion leveraging retrieved abstract reasoning templates, and (3) model-level fusion via distillation from top-ranked responses. Experiments show that FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with the optimal fusion configuration varying across benchmarks, highlighting the promise of multi-LLM log data as a practical foundation for fusing diverse LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22582v2">Fine-Grained Detection of Context-Grounded Hallucinations Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Context-grounded hallucinations are cases where model outputs contain information not verifiable against the source text. We study the applicability of LLMs for localizing such hallucinations, as a more practical alternative to existing complex evaluation pipelines. In the absence of established benchmarks for meta-evaluation of hallucinations localization, we construct one tailored to LLMs, involving a challenging human annotation of over 1,000 examples. We complement the benchmark with an LLM-based evaluation protocol, verifying its quality in a human evaluation. Since existing representations of hallucinations limit the types of errors that can be expressed, we propose a new representation based on free-form textual descriptions, capturing the full range of possible errors. We conduct a comprehensive study, evaluating four large-scale LLMs, which highlights the benchmark's difficulty, as the best model achieves an F1 score of only 0.67. Through careful analysis, we offer insights into optimal prompting strategies for the task and identify the main factors that make it challenging for LLMs: (1) a tendency to incorrectly flag missing details as inconsistent, despite being instructed to check only facts in the output; and (2) difficulty with outputs containing factually correct information absent from the source - and thus not verifiable - due to alignment with the model's parametric knowledge.
    </details>
</div>
