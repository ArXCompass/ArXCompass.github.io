# llm - 2025_10

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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- Part 19

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v1">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 82 pages, 26 figures, 34 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., ``Generate 5 jokes about coffee and their corresponding probabilities''). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01164v1">Social Welfare Function Leaderboard: When LLM Agents Allocate Social Welfare</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly entrusted with high-stakes decisions that affect human welfare. However, the principles and values that guide these models when distributing scarce societal resources remain largely unexamined. To address this, we introduce the Social Welfare Function (SWF) Benchmark, a dynamic simulation environment where an LLM acts as a sovereign allocator, distributing tasks to a heterogeneous community of recipients. The benchmark is designed to create a persistent trade-off between maximizing collective efficiency (measured by Return on Investment) and ensuring distributive fairness (measured by the Gini coefficient). We evaluate 20 state-of-the-art LLMs and present the first leaderboard for social welfare allocation. Our findings reveal three key insights: (i) A model's general conversational ability, as measured by popular leaderboards, is a poor predictor of its allocation skill. (ii) Most LLMs exhibit a strong default utilitarian orientation, prioritizing group productivity at the expense of severe inequality. (iii) Allocation strategies are highly vulnerable, easily perturbed by output-length constraints and social-influence framing. These results highlight the risks of deploying current LLMs as societal decision-makers and underscore the need for specialized benchmarks and targeted alignment for AI governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01135v1">Prompt Curriculum Learning for Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We introduce Prompt Curriculum Learning (PCL), a lightweight reinforcement learning (RL) algorithm that selects intermediate-difficulty prompts using a learned value model to post-train language models. Since post-training LLMs via RL remains sensitive to batching and prompt selection strategies, we first conduct a series of systematic experiments where we (1) determine the optimal training batch size that balances generation efficiency and gradient quality and (2) establish the importance of focusing on prompts of intermediate difficulty for the policy. We build upon these results to design PCL, which identifies prompts of intermediate difficulty for the current policy in an on-policy manner by using a value model that is concurrently updated based on the current policy. By focusing on informative prompts that yield high effective ratios, PCL achieves either the highest performance or requires significantly less time to reach comparable performance to its counterparts. Compared to rollout-based filtering methods, PCL avoids costly rollouts and achieves $12.1\times$ and $16.9\times$ faster speed on identifying intermediate-difficulty prompts when training on MATH and DeepScaleR, respectively. We further demonstrate that our value model accurately predicts prompt difficulty and allows PCL to focus on progressively more challenging prompts during RL. Our results present a new methodology that delivers improved tradeoff between upper-bound performance and efficiency for reasoning-focused RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01123v1">Rethinking Thinking Tokens: LLMs as Improvement Operators</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Reasoning training incentivizes LLMs to produce long chains of thought (long CoT), which among other things, allows them to explore solution strategies with self-checking. This results in higher accuracy, but inflates context length, token/compute cost, and answer latency. We ask: Can current models leverage their metacognition to provide other combinations on this Pareto frontier, e.g., better accuracy with lower context length and/or latency? Abstractly, we view the model as an improvement operator on its own "thoughts" with a continuum of possible strategies. We identify an interesting inference family Parallel-Distill-Refine (PDR), which performs the following: (i) generate diverse drafts in parallel; (ii) distill them into a bounded, textual workspace; and (iii) refine conditioned on this workspace, producing an output that seeds the next round. Importantly, context length (hence compute cost) is controllable via degree of parallelism, and is no longer conflated with the total number of generated tokens. We report PDR instantiations of current models that give better accuracy than long CoT while incurring lower latency. Setting degree of parallelism to 1 yields an interesting subcase, Sequential Refinement (SR) (iteratively improve a single candidate answer) which provides performance superior to long CoT. Success of such model orchestrations raises the question whether further training could shift the Pareto frontier. To this end, we train an 8B thinking model with Reinforcement Learning (RL) to make it consistent with PDR as the inference method. On math tasks with verifiable answers, iterative pipelines surpass single-pass baselines at matched sequential budgets, with PDR delivering the largest gains (e.g., +11% on AIME 2024 and +9% on AIME 2025).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01111v1">Augmenting LLMs for General Time Series Understanding and Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Time series data is fundamental to decision-making in many crucial domains including healthcare, finance, and environmental science. However, analyzing this data often requires incorporating unstructured contextual information, answering domain-specific questions, and generating natural language explanations -- capabilities that traditional time series models lack due to their inability to process text. While Large Language Models (LLMs) excel at contextual reasoning and knowledge integration, they struggle with numerical time series due to inefficient text-based representations and limited exposure to temporal data during pretraining. We address this gap by augmenting an LLM with specialized time series perception through a patch-based encoder-decoder architecture. We train this Time Series-augmented LLM (TsLLM) on a large corpus of over 2 million interleaved time series and text examples spanning diverse analysis tasks: forecasting with contextual information, time series question-answering, pattern explanation, classification with natural language outputs, and report generation. This training enables TsLLM to leverage both its language understanding and newly acquired temporal reasoning capabilities. While not designed to surpass specialized models on traditional benchmarks, TsLLM demonstrates strong performance on tasks requiring the integration of time series analysis with natural language -- capabilities that existing approaches cannot provide. Our work establishes a new paradigm for time series analysis that bridges numerical computation and natural language understanding, democratizing access to sophisticated temporal reasoning through natural language interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01088v1">Safety Instincts: LLMs Learn to Trust Their Internal Compass for Self-Defense</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Ensuring Large Language Model (LLM) safety remains challenging due to the absence of universal standards and reliable content validators, making it difficult to obtain effective training signals. We discover that aligned models already possess robust internal safety beliefs: they consistently produce high-confidence refusals to harmful requests while exhibiting high entropy when generating potentially dangerous content. This entropy gap reveals an untapped signal--models intrinsically "know" when to refuse. We introduce Safety Instincts Reinforcement Learning (SIRL), which transforms this internal confidence into a self-generated reward signal, eliminating dependence on external validators or human annotations. SIRL teaches models to trust their safety instincts by reinforcing low-entropy refusal behaviors. Evaluated on Llama and Qwen models, SIRL maintains 89%+ Defense Success Rates (DSRs) against 20+ jailbreak methods, from static prompts to adaptive attacks. Using only 15,000 unlabeled prompts, SIRL surpasses resource-intensive supervised methods while preserving performance on mathematics, coding, and conversation benchmarks. Our work demonstrates that effective alignment can emerge from within, paving the way for more autonomous and robust AI safety mechanisms that scale without extensive human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01051v1">GEM: A Gym for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which -- unlike GRPO -- is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01037v1">CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 25 pages, 10 Figures
    </div>
    <details class="paper-abstract">
      Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by \textbf{+3.30} points and \textbf{+4.82} points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01032v1">Meaningless Tokens, Meaningful Gains: How Activation Shifts Enhance LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Motivated by the puzzling observation that inserting long sequences of meaningless tokens before the query prompt can consistently enhance LLM reasoning performance, this work analyzes the underlying mechanism driving this phenomenon and based on these insights proposes a more principled method that allows for similar performance gains. First, we find that the improvements arise from a redistribution of activations in the LLM's MLP layers, where near zero activations become less frequent while large magnitude activations increase. This redistribution enhances the model's representational capacity by suppressing weak signals and promoting stronger, more informative ones. Building on this insight, we propose the Activation Redistribution Module (ARM), a lightweight inference-time technique that modifies activations directly without altering the input sequence. ARM adaptively identifies near-zero activations after the non-linear function and shifts them outward, implicitly reproducing the beneficial effects of meaningless tokens in a controlled manner. Extensive experiments across diverse benchmarks and model architectures clearly show that ARM consistently improves LLM performance on reasoning tasks while requiring only a few lines of simple code to implement. Our findings deliver both a clear mechanistic explanation for the unexpected benefits of meaningless tokens and a simple yet effective technique that harnesses activation redistribution to further improve LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01030v1">Uncovering the Computational Ingredients of Human-Like Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The ability to translate diverse patterns of inputs into structured patterns of behavior has been thought to rest on both humans' and machines' ability to learn robust representations of relevant concepts. The rapid advancement of transformer-based large language models (LLMs) has led to a diversity of computational ingredients -- architectures, fine tuning methods, and training datasets among others -- but it remains unclear which of these ingredients are most crucial for building models that develop human-like representations. Further, most current LLM benchmarks are not suited to measuring representational alignment between humans and models, making benchmark scores unreliable for assessing if current LLMs are making progress towards becoming useful cognitive models. We address these limitations by first evaluating a set of over 70 models that widely vary in their computational ingredients on a triplet similarity task, a method well established in the cognitive sciences for measuring human conceptual representations, using concepts from the THINGS database. Comparing human and model representations, we find that models that undergo instruction-finetuning and which have larger dimensionality of attention heads are among the most human aligned, while multimodal pretraining and parameter size have limited bearing on alignment. Correlations between alignment scores and scores on existing benchmarks reveal that while some benchmarks (e.g., MMLU) are better suited than others (e.g., MUSR) for capturing representational alignment, no existing benchmark is capable of fully accounting for the variance of alignment scores, demonstrating their insufficiency in capturing human-AI alignment. Taken together, our findings help highlight the computational ingredients most essential for advancing LLMs towards models of human conceptual representation and address a key benchmarking gap in LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00967v1">QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Designing and optimizing task-specific quantum circuits are crucial to leverage the advantage of quantum computing. Recent large language model (LLM)-based quantum circuit generation has emerged as a promising automatic solution. However, the fundamental challenges remain unaddressed: (i) parameterized quantum gates require precise numerical values for optimal performance, which also depend on multiple aspects, including the number of quantum gates, their parameters, and the layout/depth of the circuits. (ii) LLMs often generate low-quality or incorrect quantum circuits due to the lack of quantum domain-specific knowledge. We propose QUASAR, an agentic reinforcement learning (RL) framework for quantum circuits generation and optimization based on tool-augmented LLMs. To align the LLM with quantum-specific knowledge and improve the generated quantum circuits, QUASAR designs (i) a quantum circuit verification approach with external quantum simulators and (ii) a sophisticated hierarchical reward mechanism in RL training. Extensive evaluation shows improvements in both syntax and semantic performance of the generated quantum circuits. When augmenting a 4B LLM, QUASAR has achieved the validity of 99.31% in Pass@1 and 100% in Pass@10, outperforming industrial LLMs of GPT-4o, GPT-5 and DeepSeek-V3 and several supervised-fine-tuning (SFT)-only and RL-only baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00962v1">Analyzing Dialectical Biases in LLMs for Knowledge and Reasoning Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 EMNLP Findings 2025, 12 pages, 11 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are ubiquitous in modern day natural language processing. However, previous work has shown degraded LLM performance for under-represented English dialects. We analyze the effects of typifying "standard" American English language questions as non-"standard" dialectal variants on multiple choice question answering tasks and find up to a 20% reduction in accuracy. Additionally, we investigate the grammatical basis of under-performance in non-"standard" English questions. We find that individual grammatical rules have varied effects on performance, but some are more consequential than others: three specific grammar rules (existential "it", zero copula, and y'all) can explain the majority of performance degradation observed in multiple dialects. We call for future work to investigate bias mitigation methods focused on individual, high-impact grammatical structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00932v1">Opal: A Modular Framework for Optimizing Performance using Analytics and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 12 pages and 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise for automated code optimization but struggle without performance context. This work introduces Opal, a modular framework that connects performance analytics insights with the vast body of published by guiding LLMs to generate informed, trustworthy optimizations. Unlike traditional performance tools that identify bottlenecks but stop short of actionable suggestions, Opal bridges this long-standing gap by linking dynamic insights from hardware counters and Roofline analysis to stall events to optimization decisions. We evaluate Opal across 1640 experiments on real-world GPU kernels and find that in over 98.5% of cases, even a single insight source yields speedups, ranging on average from 19.34% to 52.3%. Our prompt template produced correct code in all but one case, where a vague diagnostic caused an unsafe suggestion. By automatically optimizing GPU kernels using performance analytics and LLMs, Opal marks a leap toward democratizing expert-level performance engineering for all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00911v1">RiskPO: Risk-based Policy Optimization via Verifiable Reward for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable reward has recently emerged as a central paradigm for post-training large language models (LLMs); however, prevailing mean-based methods, such as Group Relative Policy Optimization (GRPO), suffer from entropy collapse and limited reasoning gains. We argue that these issues stem from overemphasizing high-probability output sequences while neglecting rare but informative reasoning paths. To address these challenges, we propose Risk-based Policy Optimization (RiskPO), which substitutes classical mean-based objectives with principled risk measures. Specifically, we introduce a Mixed Value-at-Risk objective that integrates weighted attention over multiple regions of the reward distribution, thereby amplifying gradient signals on challenging instances and preventing overconfident convergence. We further design a bundling scheme that aggregates multiple questions into bundles, thus enriching the feedback signal and yielding more stable and informative training dynamics. Theoretically, we prove that the risk-averse update alleviates entropy collapse and promotes exploration. Numerically, RiskPO achieves consistent and significant improvements in mathematical reasoning, multi-modal reasoning, and code generation benchmarks, surpassing GRPO and its variants on both Pass@1 and Pass@k metrics. Our results demonstrate that risk-based optimization provides a rigorous and effective paradigm for enhancing LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00908v1">Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. Research in this area has typically framed the task as monolingual retrieval augmented by translation, treating retrieval methods and cross-lingual capabilities in isolation. Both monolingual and cross-lingual retrieval usually follow a pipeline of query expansion, ranking, re-ranking and, increasingly, question answering. Recent advances, however, have shifted from translation-based methods toward embedding-based approaches and leverage multilingual large language models (LLMs), for which aligning representations across languages remains a central challenge. The emergence of cross-lingual embeddings and multilingual LLMs has introduced a new paradigm, offering improved retrieval performance and enabling answer generation. This survey provides a comprehensive overview of developments from early translation-based methods to state-of-the-art embedding-driven and generative techniques. It presents a structured account of core CLIR components, evaluation practices, and available resources. Persistent challenges such as data imbalance and linguistic variation are identified, while promising directions are suggested for advancing equitable and effective cross-lingual information retrieval. By situating CLIR within the broader landscape of information retrieval and multilingual language processing, this work not only reviews current capabilities but also outlines future directions for building retrieval systems that are robust, inclusive, and adaptable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00881v1">Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted at ASE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into software engineering (SE) tools for tasks that extend beyond code synthesis, including judgment under uncertainty and reasoning in ethically significant contexts. We present a fully automated framework for assessing ethical reasoning capabilities across 16 LLMs in a zero-shot setting, using 30 real-world ethically charged scenarios. Each model is prompted to identify the most applicable ethical theory to an action, assess its moral acceptability, and explain the reasoning behind their choice. Responses are compared against expert ethicists' choices using inter-model agreement metrics. Our results show that LLMs achieve an average Theory Consistency Rate (TCR) of 73.3% and Binary Agreement Rate (BAR) on moral acceptability of 86.7%, with interpretable divergences concentrated in ethically ambiguous cases. A qualitative analysis of free-text explanations reveals strong conceptual convergence across models despite surface-level lexical diversity. These findings support the potential viability of LLMs as ethical inference engines within SE pipelines, enabling scalable, auditable, and adaptive integration of user-aligned ethical reasoning. Our focus is the Ethical Interpreter component of a broader profiling pipeline: we evaluate whether current LLMs exhibit sufficient interpretive stability and theory-consistent reasoning to support automated profiling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00861v1">Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00857v1">ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at https://github.com/technion-cs-nlp/ManagerBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00844v1">Learning Compact Representations of LLM Abilities via Item Response Theory</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a surge in the number of large language models (LLMs), yet efficiently managing and utilizing these vast resources remains a significant challenge. In this work, we explore how to learn compact representations of LLM abilities that can facilitate downstream tasks, such as model routing and performance prediction on new benchmarks. We frame this problem as estimating the probability that a given model will correctly answer a specific query. Inspired by the item response theory (IRT) in psychometrics, we model this probability as a function of three key factors: (i) the model's multi-skill ability vector, (2) the query's discrimination vector that separates models of differing skills, and (3) the query's difficulty scalar. To learn these parameters jointly, we introduce a Mixture-of-Experts (MoE) network that couples model- and query-level embeddings. Extensive experiments demonstrate that our approach leads to state-of-the-art performance in both model routing and benchmark accuracy prediction. Moreover, analysis validates that the learned parameters encode meaningful, interpretable information about model capabilities and query characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00841v1">LLM Routing with Dueling Feedback</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We study LLM routing, the problem of selecting the best model for each query while balancing user satisfaction, model expertise, and inference cost. We formulate routing as contextual dueling bandits, learning from pairwise preference feedback rather than absolute scores, thereby yielding label-efficient and dynamic adaptation. Building on this formulation, we introduce Category-Calibrated Fine-Tuning (CCFT), a representation-learning method that derives model embeddings from offline data using contrastive fine-tuning with categorical weighting. These embeddings enable the practical instantiation of Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB), a theoretically grounded posterior-sampling algorithm. We propose four variants of the categorical weighting that explicitly integrate model quality and cost, and we empirically evaluate the proposed methods on the RouterBench and MixInstruct datasets. Across both benchmarks, our methods achieve lower cumulative regret and faster convergence, with better robustness and performance-cost balance than strong baselines built with a general-purpose OpenAI embedding model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00829v1">Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00819v1">Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Reinforcement Learning, particularly through policy gradient methods, has played a central role in enabling reasoning capabilities of Large Language Models. However, the optimization stability of policy gradients in this setting remains understudied. As a result, existing implementations often resort to conservative hyperparameter choices to ensure stability, which requires more training samples and increases computational costs. Hence, developing models for reliably tracking the underlying optimization dynamics and leveraging them into training enables more sample-efficient regimes and further unleashes scalable post-training. We address this gap by formalizing the stochastic optimization problem of policy gradients with explicit consideration of second-order geometry. We propose a tractable computational framework that tracks and leverages curvature information during policy updates. We further employ this framework to design interventions in the optimization process through data selection. The resultant algorithm, Curvature-Aware Policy Optimization (CAPO), identifies samples that contribute to unstable updates and masks them out. Theoretically, we establish monotonic improvement guarantees under realistic assumptions. On standard math reasoning benchmarks, we empirically show that CAPO ensures stable updates under aggressive learning regimes where baselines catastrophically fail. With minimal intervention (rejecting fewer than 8% of tokens), CAPO achieves up to 30x improvement in sample efficiency over standard GRPO for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00810v1">Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We investigate how to adapt small, efficient LLMs to Faroese, a low-resource North Germanic language. Starting from English models, we continue pre-training on related Scandinavian languages, either individually or combined via merging, before fine-tuning on Faroese. We compare full fine-tuning with parameter-efficient tuning using LoRA, evaluating their impact on both linguistic accuracy and text comprehension. Due to the lack of existing Faroese evaluation data, we construct two new minimal-pair benchmarks from adapted and newly collected datasets and complement them with human evaluations by Faroese linguists. Our results demonstrate that transfer from related languages is crucial, though the optimal source language depends on the task: Icelandic enhances linguistic accuracy, whereas Danish boosts comprehension. Similarly, the choice between full fine-tuning and LoRA is task-dependent: LoRA improves linguistic acceptability and slightly increases human evaluation scores on the base model, while full fine-tuning yields stronger comprehension performance and better preserves model capabilities during downstream fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00777v1">In-Place Feedback: A New Paradigm for Guiding LLMs in Multi-Turn Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 28 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly studied in the context of multi-turn reasoning, where models iteratively refine their outputs based on user-provided feedback. Such settings are crucial for tasks that require complex reasoning, yet existing feedback paradigms often rely on issuing new messages. LLMs struggle to integrate these reliably, leading to inconsistent improvements. In this work, we introduce in-place feedback, a novel interaction paradigm in which users directly edit an LLM's previous response, and the model conditions on this modified response to generate its revision. Empirical evaluations on diverse reasoning-intensive benchmarks reveal that in-place feedback achieves better performance than conventional multi-turn feedback while using $79.1\%$ fewer tokens. Complementary analyses on controlled environments further demonstrate that in-place feedback resolves a core limitation of multi-turn feedback: models often fail to apply feedback precisely to erroneous parts of the response, leaving errors uncorrected and sometimes introducing new mistakes into previously correct content. These findings suggest that in-place feedback offers a more natural and effective mechanism for guiding LLMs in reasoning-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00662v1">Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations.
    </details>
</div>
