# llm - 2025_06

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15167v1">LLM Agent for Hyper-Parameter Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 6 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Hyper-parameters are essential and critical for the performance of communication algorithms. However, current hyper-parameters tuning methods for warm-start particles swarm optimization with cross and mutation (WS-PSO-CM) algortihm for radio map-enabled unmanned aerial vehicle (UAV) trajectory and communication are primarily heuristic-based, exhibiting low levels of automation and unsatisfactory performance. In this paper, we design an large language model (LLM) agent for automatic hyper-parameters-tuning, where an iterative framework and model context protocol (MCP) are applied. In particular, the LLM agent is first setup via a profile, which specifies the mission, background, and output format. Then, the LLM agent is driven by the prompt requirement, and iteratively invokes WS-PSO-CM algorithm for exploration. Finally, the LLM agent autonomously terminates the loop and returns a set of hyper-parameters. Our experiment results show that the minimal sum-rate achieved by hyper-parameters generated via our LLM agent is significantly higher than those by both human heuristics and random generation methods. This indicates that an LLM agent with PSO knowledge and WS-PSO-CM algorithm background is useful in finding high-performance hyper-parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15155v1">eLLM: Elastic Memory Management Framework for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly being deployed in datacenters. Serving these models requires careful memory management, as their memory usage includes static weights, dynamic activations, and key-value caches. While static weights are constant and predictable, dynamic components such as activations and KV caches change frequently during runtime, presenting significant challenges for efficient memory management. Modern LLM serving systems typically handle runtime memory and KV caches at distinct abstraction levels: runtime memory management relies on static tensor abstractions, whereas KV caches utilize a page table-based virtualization layer built on top of the tensor abstraction. This virtualization dynamically manages KV caches to mitigate memory fragmentation. However, this dual-level approach fundamentally isolates runtime memory and KV cache management, resulting in suboptimal memory utilization under dynamic workloads, which can lead to a nearly 20% drop in throughput. To address these limitations, we propose eLLM, an elastic memory management framework inspired by the classical memory ballooning mechanism in operating systems. The core components of eLLM include: (1) Virtual Tensor Abstraction, which decouples the virtual address space of tensors from the physical GPU memory, creating a unified and flexible memory pool; (2) an Elastic Memory Mechanism that dynamically adjusts memory allocation through runtime memory inflation and deflation, leveraging CPU memory as an extensible buffer; and (3) a Lightweight Scheduling Strategy employing SLO-aware policies to optimize memory utilization and effectively balance performance trade-offs under stringent SLO constraints. Comprehensive evaluations demonstrate that eLLM significantly outperforms state-of-the-art systems, 2.32x higher decoding throughput, and supporting 3x larger batch sizes for 128K-token inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03997v2">YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 accepted to International Conference on Intelligent Robots and Systems (IROS2025)
    </div>
    <details class="paper-abstract">
      Advancements in deep multi-agent reinforcement learning (MARL) have positioned it as a promising approach for decision-making in cooperative games. However, it still remains challenging for MARL agents to learn cooperative strategies for some game environments. Recently, large language models (LLMs) have demonstrated emergent reasoning capabilities, making them promising candidates for enhancing coordination among the agents. However, due to the model size of LLMs, it can be expensive to frequently infer LLMs for actions that agents can take. In this work, we propose You Only LLM Once for MARL (YOLO-MARL), a novel framework that leverages the high-level task planning capabilities of LLMs to improve the policy learning process of multi-agents in cooperative games. Notably, for each game environment, YOLO-MARL only requires one time interaction with LLMs in the proposed strategy generation, state interpretation and planning function generation modules, before the MARL policy training process. This avoids the ongoing costs and computational time associated with frequent LLMs API calls during training. Moreover, trained decentralized policies based on normal-sized neural networks operate independently of the LLM. We evaluate our method across two different environments and demonstrate that YOLO-MARL outperforms traditional MARL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01203v2">HetGCoT: Heterogeneous Graph-Enhanced Chain-of-Thought LLM Reasoning for Academic Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Academic question answering (QA) in heterogeneous scholarly networks presents unique challenges requiring both structural understanding and interpretable reasoning. While graph neural networks (GNNs) capture structured graph information and large language models (LLMs) demonstrate strong capabilities in semantic comprehension, current approaches lack integration at the reasoning level. We propose HetGCoT, a framework enabling LLMs to effectively leverage and learn information from graphs to reason interpretable academic QA results. Our framework introduces three technical contributions: (1) a framework that transforms heterogeneous graph structural information into LLM-processable reasoning chains, (2) an adaptive metapath selection mechanism identifying relevant subgraphs for specific queries, and (3) a multi-step reasoning strategy systematically incorporating graph contexts into the reasoning process. Experiments on OpenAlex and DBLP datasets show our approach outperforms all sota baselines. The framework demonstrates adaptability across different LLM architectures and applicability to various scholarly question answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15131v1">Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06809v3">Root Defence Strategies: Ensuring Safety of LLM at the Decoding Level</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated immense utility across various industries. However, as LLMs advance, the risk of harmful outputs increases due to incorrect or malicious instruction prompts. While current methods effectively address jailbreak risks, they share common limitations: 1) Judging harmful responses from the prefill-level lacks utilization of the model's decoding outputs, leading to relatively lower effectiveness and robustness. 2) Rejecting potentially harmful responses based on a single evaluation can significantly impair the model's helpfulness.This paper examines the LLMs' capability to recognize harmful outputs, revealing and quantifying their proficiency in assessing the danger of previous tokens. Motivated by pilot experiment results, we design a robust defense mechanism at the decoding level. Our novel decoder-oriented, step-by-step defense architecture corrects harmful queries directly rather than rejecting them outright. We introduce speculative decoding to enhance usability and facilitate deployment to boost secure decoding speed. Extensive experiments demonstrate that our approach improves model security without compromising reasoning speed. Notably, our method leverages the model's ability to discern hazardous information, maintaining its helpfulness compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15076v1">Learning-Time Encoding Shapes Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in the real world, the ability to ``unlearn'', or remove specific pieces of knowledge post hoc, has become essential for a variety of reasons ranging from privacy regulations to correcting outdated or harmful content. Prior work has proposed unlearning benchmarks and algorithms, and has typically assumed that the training process and the target model are fixed. In this work, we empirically investigate how learning-time choices in knowledge encoding impact the effectiveness of unlearning factual knowledge. Our experiments reveal two key findings: (1) learning with paraphrased descriptions improves unlearning performance and (2) unlearning individual piece of knowledge from a chunk of text is challenging. Our results suggest that learning-time knowledge encoding may play a central role in enabling reliable post-hoc unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16205v6">LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has brought impressive advancements across various tasks. However, despite these achievements, LLMs still pose inherent safety risks, especially in the context of jailbreak attacks. Most existing jailbreak methods follow an input-level manipulation paradigm to bypass safety mechanisms. Yet, as alignment techniques improve, such attacks are becoming increasingly detectable. In this work, we identify an underexplored threat vector: the model's internal reasoning process, which can be manipulated to elicit harmful outputs in a more stealthy way. To explore this overlooked attack surface, we propose a novel black-box jailbreak attack method, Analyzing-based Jailbreak (ABJ). ABJ comprises two independent attack paths: textual and visual reasoning attacks, which exploit the model's multimodal reasoning capabilities to bypass safety mechanisms, comprehensively exposing vulnerabilities in its reasoning chain. We conduct extensive experiments on ABJ across various open-source and closed-source LLMs, VLMs, and RLMs. In particular, ABJ achieves high attack success rate (ASR) (82.1% on GPT-4o-2024-11-20) with exceptional attack efficiency (AE) among all target models, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our work reveals a new type of safety risk and highlights the urgent need to mitigate implicit vulnerabilities in the model's reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15066v1">ChatModel: Automating Reference Model Design and Verification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      As the complexity of integrated circuit designs continues to escalate, the functional verification becomes increasingly challenging. Reference models, critical for accelerating the verification process, are themselves becoming more intricate and time-consuming to develop. Despite the promise shown by large language models (LLMs) in code programming, effectively generating complex reference models remains a significant hurdle. To address these challenges, we introduce ChatModel, the first LLM-aided agile reference model generation and verification platform. ChatModel streamlines the transition from design specifications to fully functional reference models by integrating design standardization and hierarchical agile modeling. Employing a building-block generation strategy, it not only enhances the design capabilities of LLMs for reference models but also significantly boosts verification efficiency. We evaluated ChatModel on 300 designs of varying complexity, demonstrating substantial improvements in both efficiency and quality of reference model generation. ChatModel achieved a peak performance improvement of 55.02% compared to alternative methods, with notable enhancements in generation stability, and delivered a 9.18x increase in its capacity to produce reference model designs. Furthermore, it accelerated the iterative process of reference model design and validation by an average of 5.90x compared to traditional approaches. These results highlight the potential of ChatModel to significantly advance the automation of reference model generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14397v2">Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Negation is a fundamental linguistic phenomenon that poses persistent challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Existing benchmarks often treat negation as a side case within broader tasks like natural language inference, resulting in a lack of benchmarks that exclusively target negation understanding. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly designed to assess sentence-level negation understanding in LLMs. Thunder-NUBench goes beyond surface-level cue detection by contrasting standard negation with structurally diverse alternatives such as local negation, contradiction, and paraphrase. The benchmark consists of manually curated sentence-negation pairs and a multiple-choice dataset that enables in-depth evaluation of models' negation understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15911v1">From RAG to Agentic: Validating Islamic-Medicine Responses with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Under-review at the 4th Muslims in Machine Learning (MusIML) Workshop (ICML-25)
    </div>
    <details class="paper-abstract">
      Centuries-old Islamic medical texts like Avicenna's Canon of Medicine and the Prophetic Tibb-e-Nabawi encode a wealth of preventive care, nutrition, and holistic therapies, yet remain inaccessible to many and underutilized in modern AI systems. Existing language-model benchmarks focus narrowly on factual recall or user preference, leaving a gap in validating culturally grounded medical guidance at scale. We propose a unified evaluation pipeline, Tibbe-AG, that aligns 30 carefully curated Prophetic-medicine questions with human-verified remedies and compares three LLMs (LLaMA-3, Mistral-7B, Qwen2-7B) under three configurations: direct generation, retrieval-augmented generation, and a scientific self-critique filter. Each answer is then assessed by a secondary LLM serving as an agentic judge, yielding a single 3C3H quality score. Retrieval improves factual accuracy by 13%, while the agentic prompt adds another 10% improvement through deeper mechanistic insight and safety considerations. Our results demonstrate that blending classical Islamic texts with retrieval and self-evaluation enables reliable, culturally sensitive medical question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17264v4">Medha: Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) handle increasingly longer contexts, serving long inference requests of millions of tokens presents unique challenges. We show that existing work for long context inference is largely based on techniques from long context training, and does not handle the high variability in input lengths during inference. This leads to inefficient resource utilization, server fragmentation, and head-of-line (HOL) blocking. We present Medha, an end-to-end system for efficient long-context LLM inference that addresses these challenges through fine-grained time sharing. Medha introduces three key innovations: (1) the mechanism of adaptive prefill chunking to help mitigate HOL blocking with preemption; (2) two new parallelism strategies: Sequence Pipeline Parallelism (SPP) to reduce time-to-first-token by pipelining prefill chunks, and KV-Cache Parallelism (KVP) to lower time-peroutput-token by distributing decoding across servers; and (3) a novel input-length aware least remaining slack scheduling to meet Service Level Objectives (SLOs). Medha enables exact inference scaling beyond 10 million tokens, maintaining high throughput and low latency across mixed-length workloads. Compared to state-of-the-art systems, Medha reduces server fragmentation, cuts median latency by up to 30x, and improves throughput by over 5x, delivering production-scale long-context inference without compromising performance on shorter requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22998v2">LLM Agents for Bargaining with Utility-based Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Bargaining, a critical aspect of real-world interactions, presents challenges for large language models (LLMs) due to limitations in strategic depth and adaptation to complex human factors. Existing benchmarks often fail to capture this real-world complexity. To address this and enhance LLM capabilities in realistic bargaining, we introduce a comprehensive framework centered on utility-based feedback. Our contributions are threefold: (1) BargainArena, a novel benchmark dataset with six intricate scenarios (e.g., deceptive practices, monopolies) to facilitate diverse strategy modeling; (2) human-aligned, economically-grounded evaluation metrics inspired by utility theory, incorporating agent utility and negotiation power, which implicitly reflect and promote opponent-aware reasoning (OAR); and (3) a structured feedback mechanism enabling LLMs to iteratively refine their bargaining strategies. This mechanism can positively collaborate with in-context learning (ICL) prompts, including those explicitly designed to foster OAR. Experimental results show that LLMs often exhibit negotiation strategies misaligned with human preferences, and that our structured feedback mechanism significantly improves their performance, yielding deeper strategic and opponent-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15866v1">Understanding Online Polarization Through Human-Agent Interaction in a Synthetic LLM-Based Social Network</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Accepted for publication in the Proceedings of the Nineteenth International AAAI Conference on Web and Social Media (ICWSM 2025). This is the authors' version of the work, with corrections to table cross-references. The definitive Version of Record is available at https://doi.org/10.1609/icwsm.v19i1.35826. arXiv admin note: substantial text overlap with arXiv:2502.01340
    </div>
    <details class="paper-abstract">
      The rise of social media has fundamentally transformed how people engage in public discourse and form opinions. While these platforms offer unprecedented opportunities for democratic engagement, they have been implicated in increasing social polarization and the formation of ideological echo chambers. Previous research has primarily relied on observational studies of social media data or theoretical modeling approaches, leaving a significant gap in our understanding of how individuals respond to and are influenced by polarized online environments. Here we present a novel experimental framework for investigating polarization dynamics that allows human users to interact with LLM-based artificial agents in a controlled social network simulation. Through a user study with 122 participants, we demonstrate that this approach can successfully reproduce key characteristics of polarized online discourse while enabling precise manipulation of environmental factors. Our results provide empirical validation of theoretical predictions about online polarization, showing that polarized environments significantly increase perceived emotionality and group identity salience while reducing expressed uncertainty. These findings extend previous observational and theoretical work by providing causal evidence for how specific features of online environments influence user perceptions and behaviors. More broadly, this research introduces a powerful new methodology for studying social media dynamics, offering researchers unprecedented control over experimental conditions while maintaining ecological validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20620v2">Rectifying Belief Space via Unlearning to Harness LLMs' Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Accepted at ACL2025 Findings (long)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can exhibit advanced reasoning yet still generate incorrect answers. We hypothesize that such errors frequently stem from spurious beliefs, propositions the model internally considers true but are incorrect. To address this, we propose a method to rectify the belief space by suppressing these spurious beliefs while simultaneously enhancing true ones, thereby enabling more reliable inferences. Our approach first identifies the beliefs that lead to incorrect or correct answers by prompting the model to generate textual explanations, using our Forward-Backward Beam Search (FBBS). We then apply unlearning to suppress the identified spurious beliefs and enhance the true ones, effectively rectifying the model's belief space. Empirical results on multiple QA datasets and LLMs show that our method corrects previously misanswered questions without harming overall model performance. Furthermore, our approach yields improved generalization on unseen data, suggesting that rectifying a model's belief space is a promising direction for mitigating errors and enhancing overall reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14448v1">How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14429v1">LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 16 pages, 12 figures, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably \textbf{\textit{stable perplexity}} during direct context extrapolation. Furthermore, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct \textbf{\textit{local perception}} phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first context extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14387v1">Don't Make It Up: Preserving Ignorance Awareness in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Existing work on mitigating catastrophic forgetting in large language model (LLM) fine-tuning has primarily focused on preserving specific data or tasks, while critically overlooking the degradation of essential capabilities instilled through safety alignment, particularly the model's ability to faithfully express ignorance. In this work, we show that this capability is significantly degraded during conventional fine-tuning, leading to undesired behaviors such as hallucinations. To address this novel but highly practical problem, we propose SEAT, a simple and effective fine-tuning approach that preserves both fine-tuning performance and the model's inherent ability to acknowledge its ignorance. SEAT integrates two key components: (1) sparse training that constrains activation drift, and (2) a novel entity perturbation method with KL-divergence regularization, designed to counter knowledge entanglement. Experimental results demonstrate that SEAT significantly outperforms baselines in preserving ignorance awareness while retaining fine-tuning performance, offering a more robust solution for LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14374v1">Excessive Reasoning Attack on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling. However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking). In this work, we expose a novel threat: adversarial inputs can be crafted to exploit excessive reasoning behaviors and substantially increase computational overhead without compromising model utility. Therefore, we propose a novel loss framework consisting of three components: (1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs; (2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and (3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs. We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. Empirical results demonstrate a 3x to 9x increase in reasoning length with comparable utility performance. Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, o1-mini, DeepSeek-R1, and QWQ models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14371v1">ELLIS Alicante at CQs-Gen 2025: Winning the critical thinking questions shared task: LLM-based question generation and selection</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Proceedings of the 12th Workshop on Argument Mining
    </div>
    <details class="paper-abstract">
      The widespread adoption of chat interfaces based on Large Language Models (LLMs) raises concerns about promoting superficial learning and undermining the development of critical thinking skills. Instead of relying on LLMs purely for retrieving factual information, this work explores their potential to foster deeper reasoning by generating critical questions that challenge unsupported or vague claims in debate interventions. This study is part of a shared task of the 12th Workshop on Argument Mining, co-located with ACL 2025, focused on automatic critical question generation. We propose a two-step framework involving two small-scale open source language models: a Questioner that generates multiple candidate questions and a Judge that selects the most relevant ones. Our system ranked first in the shared task competition, demonstrating the potential of the proposed LLM-based approach to encourage critical engagement with argumentative texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07709v3">MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Open-ended learning agents must efficiently prioritize goals in vast possibility spaces, focusing on those that maximize learning progress (LP). When such autotelic exploration is achieved by LLM agents trained with online RL in high-dimensional and evolving goal spaces, a key challenge for LP prediction is modeling one's own competence, a form of metacognitive monitoring. Traditional approaches either require extensive sampling or rely on brittle expert-defined goal groupings. We introduce MAGELLAN, a metacognitive framework that lets LLM agents learn to predict their competence and LP online. By capturing semantic relationships between goals, MAGELLAN enables sample-efficient LP estimation and dynamic adaptation to evolving goal spaces through generalization. In an interactive learning environment, we show that MAGELLAN improves LP prediction efficiency and goal prioritization, being the only method allowing the agent to fully master a large and evolving goal space. These results demonstrate how augmenting LLM agents with a metacognitive ability for LP predictions can effectively scale curriculum learning to open-ended goal spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14337v1">LLM-Powered Intent-Based Categorization of Phishing Emails</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Phishing attacks remain a significant threat to modern cybersecurity, as they successfully deceive both humans and the defense mechanisms intended to protect them. Traditional detection systems primarily focus on email metadata that users cannot see in their inboxes. Additionally, these systems struggle with phishing emails, which experienced users can often identify empirically by the text alone. This paper investigates the practical potential of Large Language Models (LLMs) to detect these emails by focusing on their intent. In addition to the binary classification of phishing emails, the paper introduces an intent-type taxonomy, which is operationalized by the LLMs to classify emails into distinct categories and, therefore, generate actionable threat information. To facilitate our work, we have curated publicly available datasets into a custom dataset containing a mix of legitimate and phishing emails. Our results demonstrate that existing LLMs are capable of detecting and categorizing phishing emails, underscoring their potential in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14336v1">AviationLLM: An LLM-based Knowledge System for Aviation Training</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Aviation training is a core link in ensuring flight safety, improving industry efficiency and promoting sustainable development. It not only involves flight simulation but also requires the learning of a great deal of professional aviation theory knowledge. In the existing training system, the knowledge is mainly imparted by the the instructors. However, the number of instructors is limited and the professional answers obtained from the Internet are not accurate enough, resulting in low training efficiency. To address this, we introduced LLM, but the basic pre-trained model cannot provide accurate answers to professional fields, so we fine-tuned it. Traditional Supervised Fine-Tuning (SFT) risk generating superficially plausible but factually incorrect responses due to insufficient data coverage. To address this, we employ Direct Preference Optimization(DPO). This paper proposes Retrieval-Augmented LLM Alignment via Direct Preference Optimization(RALA-DPO). We select open source pre-trained LLM Qwen and adapt it to aviation theory training through DPO-based domain alignment. Simultaneously, to mitigate hallucinations caused by training data biases, knowledge obsolescence, or domain knowledge gaps, we implement Retrieval-Augmented Generation(RAG) technology that combines generative and retrieval models. RALA-DPO effectively retrieves relevant information from external knowledge bases and delivers precise and high-quality responses through the generative model. Experimental results demonstrate that RALA-DPO can improve accuracy in response to professional aviation knowledge. With integrated RAG mechanisms, this system can further improve the accuracy of answers and achieve zero-cost knowledge updates simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14299v1">ADRD: LLM-Driven Autonomous Driving Based on Rule-based Decision Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      How to construct an interpretable autonomous driving decision-making system has become a focal point in academic research. In this study, we propose a novel approach that leverages large language models (LLMs) to generate executable, rule-based decision systems to address this challenge. Specifically, harnessing the strong reasoning and programming capabilities of LLMs, we introduce the ADRD(LLM-Driven Autonomous Driving Based on Rule-based Decision Systems) framework, which integrates three core modules: the Information Module, the Agents Module, and the Testing Module. The framework operates by first aggregating contextual driving scenario information through the Information Module, then utilizing the Agents Module to generate rule-based driving tactics. These tactics are iteratively refined through continuous interaction with the Testing Module. Extensive experimental evaluations demonstrate that ADRD exhibits superior performance in autonomous driving decision tasks. Compared to traditional reinforcement learning approaches and the most advanced LLM-based methods, ADRD shows significant advantages in terms of interpretability, response speed, and driving performance. These results highlight the framework's ability to achieve comprehensive and accurate understanding of complex driving scenarios, and underscore the promising future of transparent, rule-based decision systems that are easily modifiable and broadly applicable. To the best of our knowledge, this is the first work that integrates large language models with rule-based systems for autonomous driving decision-making, and our findings validate its potential for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12376v2">ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Accepted at ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: https://github.com/ulab-uiuc/consistencychecker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14245v1">Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). However, a critical paradox clouds its efficacy: RLVR-tuned models often underperform their base models on the $Pass@K$ metric for solution-finding, leading to the hypothesis that RLVR merely re-weights existing reasoning paths at the cost of reasoning diversity. In this work, we resolve this contradiction by identifying the source of the problem: the $Pass@K$ metric itself is a flawed measure of reasoning, as it credits correct final answers that probably arise from inaccurate or incomplete chains of thought (CoTs). To address this, we introduce a more precise evaluation metric, $CoT$-$Pass@K$, which mandates that both the reasoning path and the final answer be correct. We provide a new theoretical foundation that formalizes how RLVR, unlike traditional RL, is uniquely structured to incentivize logical integrity. Our empirical results are supportive: using $CoT$-$Pass@K$, we observe that RLVR can incentivize the generalization of correct reasoning for all values of $K$. Furthermore, by analyzing the training dynamics, we find that this enhanced reasoning capability emerges early in the training process and smoothly generalizes. Our work provides a clear perspective on the role of RLVR, offers a more reliable method for its evaluation, and confirms its potential to genuinely advance machine reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11368v2">GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed as autonomous agents capable of following user instructions and making decisions in real-world applications. Previous studies have made notable progress in benchmarking the instruction following capabilities of LLMs in general domains, with a primary focus on their inherent commonsense knowledge. Recently, LLMs have been increasingly deployed as domain-oriented agents, which rely on domain-oriented guidelines that may conflict with their commonsense knowledge. These guidelines exhibit two key characteristics: they consist of a wide range of domain-oriented rules and are subject to frequent updates. Despite these challenges, the absence of comprehensive benchmarks for evaluating the domain-oriented guideline following capabilities of LLMs presents a significant obstacle to their effective assessment and further development. In this paper, we introduce GuideBench, a comprehensive benchmark designed to evaluate guideline following performance of LLMs. GuideBench evaluates LLMs on three critical aspects: (i) adherence to diverse rules, (ii) robustness to rule updates, and (iii) alignment with human preferences. Experimental results on a range of LLMs indicate substantial opportunities for improving their ability to follow domain-oriented guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06926v2">Effect of Selection Format on LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      This paper investigates a critical aspect of large language model (LLM) performance: the optimal formatting of classification task options in prompts. Through an extensive experimental study, we compared two selection formats -- bullet points and plain English -- to determine their impact on model performance. Our findings suggest that presenting options via bullet points generally yields better results, although there are some exceptions. Furthermore, our research highlights the need for continued exploration of option formatting to drive further improvements in model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14852v1">Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      LLM-based agentic applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agentic applications where outputs depend on external data or environmental contexts. We propose agentic plan caching, a novel approach that extracts, stores, adapts, and reuses structured plan templates from planning stages of agentic applications across semantically similar tasks to reduce the cost of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agentic applications shows that our system can reduce costs by 46.62% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14161v1">MIST: Towards Multi-dimensional Implicit Bias and Stereotype Evaluation of LLMs via Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) in Large Language Models (LLMs) refers to their capacity for reasoning about mental states, yet failures in this capacity often manifest as systematic implicit bias. Evaluating this bias is challenging, as conventional direct-query methods are susceptible to social desirability effects and fail to capture its subtle, multi-dimensional nature. To this end, we propose an evaluation framework that leverages the Stereotype Content Model (SCM) to reconceptualize bias as a multi-dimensional failure in ToM across Competence, Sociability, and Morality. The framework introduces two indirect tasks: the Word Association Bias Test (WABT) to assess implicit lexical associations and the Affective Attribution Test (AAT) to measure covert affective leanings, both designed to probe latent stereotypes without triggering model avoidance. Extensive experiments on 8 State-of-the-Art LLMs demonstrate our framework's capacity to reveal complex bias structures, including pervasive sociability bias, multi-dimensional divergence, and asymmetric stereotype amplification, thereby providing a more robust methodology for identifying the structural nature of implicit bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14851v1">Efficient Serving of LLM Applications with Probabilistic Demand Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Applications based on Large Language Models (LLMs) contains a series of tasks to address real-world problems with boosted capability, which have dynamic demand volumes on diverse backends. Existing serving systems treat the resource demands of LLM applications as a blackbox, compromising end-to-end efficiency due to improper queuing order and backend warm up latency. We find that the resource demands of LLM applications can be modeled in a general and accurate manner with Probabilistic Demand Graph (PDGraph). We then propose Hermes, which leverages PDGraph for efficient serving of LLM applications. Confronting probabilistic demand description, Hermes applies the Gittins policy to determine the scheduling order that can minimize the average application completion time. It also uses the PDGraph model to help prewarm cold backends at proper moments. Experiments with diverse LLM applications confirm that Hermes can effectively improve the application serving efficiency, reducing the average completion time by over 70% and the P95 completion time by over 80%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02621v2">LLMs Help Alleviate the Cross-Subject Variability in Brain Signal and Language Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 The result is no longer believeable. Teaching force issue exists in the infer time of LLM
    </div>
    <details class="paper-abstract">
      Decoding human activity from EEG signals has long been a popular research topic. While recent studies have increasingly shifted focus from single-subject to cross-subject analysis, few have explored the model's ability to perform zero-shot predictions on EEG signals from previously unseen subjects. This research aims to investigate whether deep learning methods can capture subject-independent semantic information inherent in human EEG signals. Such insights are crucial for Brain-Computer Interfaces (BCI) because, on one hand, they demonstrate the model's robustness against subject-specific temporal biases, and on the other, they significantly enhance the generalizability of downstream tasks. We employ Large Language Models (LLMs) as denoising agents to extract subject-independent semantic features from noisy EEG signals. Experimental results, including ablation studies, highlight the pivotal role of LLMs in decoding subject-independent semantic information from noisy EEG data. We hope our findings will contribute to advancing BCI research and assist both academia and industry in applying EEG signals to a broader range of applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20653v3">Mind the Inconspicuous: Revealing the Hidden Weakness in Aligned LLMs' Refusal Boundaries</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 published at USENIX Security 25
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have led to impressive alignment where models learn to distinguish harmful from harmless queries through supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). In this paper, we reveal a subtle yet impactful weakness in these aligned models. We find that simply appending multiple end of sequence (eos) tokens can cause a phenomenon we call context segmentation, which effectively shifts both harmful and benign inputs closer to the refusal boundary in the hidden space. Building on this observation, we propose a straightforward method to BOOST jailbreak attacks by appending eos tokens. Our systematic evaluation shows that this strategy significantly increases the attack success rate across 8 representative jailbreak techniques and 16 open-source LLMs, ranging from 2B to 72B parameters. Moreover, we develop a novel probing mechanism for commercial APIs and discover that major providers such as OpenAI, Anthropic, and Qwen do not filter eos tokens, making them similarly vulnerable. These findings highlight a hidden yet critical blind spot in existing alignment and content filtering approaches. We call for heightened attention to eos tokens' unintended influence on model behaviors, particularly in production systems. Our work not only calls for an input-filtering based defense, but also points to new defenses that make refusal boundaries more robust and generalizable, as well as fundamental alignment techniques that can defend against context segmentation attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13510v2">Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly power applications used by children and adolescents, ensuring safe and age-appropriate interactions has become an urgent ethical imperative. Despite progress in AI safety, current evaluations predominantly focus on adults, neglecting the unique vulnerabilities of minors engaging with generative AI. We introduce Safe-Child-LLM, a comprehensive benchmark and dataset for systematically assessing LLM safety across two developmental stages: children (7-12) and adolescents (13-17). Our framework includes a novel multi-part dataset of 200 adversarial prompts, curated from red-teaming corpora (e.g., SG-Bench, HarmBench), with human-annotated labels for jailbreak success and a standardized 0-5 ethical refusal scale. Evaluating leading LLMs -- including ChatGPT, Claude, Gemini, LLaMA, DeepSeek, Grok, Vicuna, and Mistral -- we uncover critical safety deficiencies in child-facing scenarios. This work highlights the need for community-driven benchmarks to protect young users in LLM interactions. To promote transparency and collaborative advancement in ethical AI development, we are publicly releasing both our benchmark datasets and evaluation codebase at https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Benchmark.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14086v1">InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant strides across various information retrieval tasks, particularly as rerankers, owing to their strong generalization and knowledge-transfer capabilities acquired from extensive pretraining. In parallel, the rise of LLM-based chat interfaces has raised user expectations, encouraging users to pose more complex queries that necessitate retrieval by ``reasoning'' over documents rather than through simple keyword matching or semantic similarity. While some recent efforts have exploited reasoning abilities of LLMs for reranking such queries, considerable potential for improvement remains. In that regards, we introduce InsertRank, an LLM-based reranker that leverages lexical signals like BM25 scores during reranking to further improve retrieval performance. InsertRank demonstrates improved retrieval effectiveness on -- BRIGHT, a reasoning benchmark spanning 12 diverse domains, and R2MED, a specialized medical reasoning retrieval benchmark spanning 8 different tasks. We conduct an exhaustive evaluation and several ablation studies and demonstrate that InsertRank consistently improves retrieval effectiveness across multiple families of LLMs, including GPT, Gemini, and Deepseek models. %In addition, we also conduct ablation studies on normalization by varying the scale of the BM25 scores, and positional bias by shuffling the order of the documents. With Deepseek-R1, InsertRank achieves a score of 37.5 on the BRIGHT benchmark. and 51.1 on the R2MED benchmark, surpassing previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15740v1">SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as autonomous agents in complex and long horizon settings, it is critical to evaluate their ability to sabotage users by pursuing hidden objectives. We study the ability of frontier LLMs to evade monitoring and achieve harmful hidden goals while completing a wide array of realistic tasks. We evaluate a broad range of frontier LLMs using SHADE (Subtle Harmful Agent Detection & Evaluation)-Arena, the first highly diverse agent evaluation dataset for sabotage and monitoring capabilities of LLM agents. SHADE-Arena consists of complex pairs of benign main tasks and harmful side objectives in complicated environments. Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor. When measuring agent ability to (a) complete the main task, (b) complete the side task, and (c) avoid detection, we find that the best performing frontier models score 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro) as sabotage agents when overseen by Claude 3.6 Sonnet. For current frontier models, success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor. We also use SHADE-Arena to measure models' monitoring abilities, with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts. We find that for now, models still struggle at sabotage due to failures in long-context main task execution. However, our measurements already demonstrate the difficulty of monitoring for subtle sabotage attempts, which we expect to only increase in the face of more complex and longer-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17299v1">LLM Jailbreak Oracle</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly deployed in safety-critical applications, the lack of systematic methods to assess their vulnerability to jailbreak attacks presents a critical security gap. We introduce the jailbreak oracle problem: given a model, prompt, and decoding strategy, determine whether a jailbreak response can be generated with likelihood exceeding a specified threshold. This formalization enables a principled study of jailbreak vulnerabilities. Answering the jailbreak oracle problem poses significant computational challenges -- the search space grows exponentially with the length of the response tokens. We present Boa, the first efficient algorithm for solving the jailbreak oracle problem. Boa employs a three-phase search strategy: (1) constructing block lists to identify refusal patterns, (2) breadth-first sampling to identify easily accessible jailbreaks, and (3) depth-first priority search guided by fine-grained safety scores to systematically explore promising low-probability paths. Boa enables rigorous security assessments including systematic defense evaluation, standardized comparison of red team attacks, and model certification under extreme adversarial conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15025v1">Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 TD,LR: How to set the learning rate for emebdding layer in LLMs?
    </div>
    <details class="paper-abstract">
      Pretraining large language models is a costly process. To make this process more efficient, several methods have been proposed to optimize model architecture/parametrization and hardware use. On the parametrization side, $\mu P$ (Maximal Update Parametrization) parametrizes model weights and learning rate (LR) in a way that makes hyperparameters (HPs) transferable with width (embedding dimension): HPs can be tuned for a small model and used for larger models without additional tuning. While $\mu$P showed impressive results in practice, recent empirical studies have reported conflicting observations when applied to LLMs. One limitation of the theory behind $\mu$P is the fact that input dimension (vocabulary size in LLMs) is considered fixed when taking the width to infinity. This is unrealistic since vocabulary size is generally much larger than width in practice. In this work, we provide a theoretical analysis of the effect of vocabulary size on training dynamics, and subsequently show that as vocabulary size increases, the training dynamics \emph{interpolate between the $\mu$P regime and another regime that we call Large Vocab (LV) Regime}, where optimal scaling rules are different from those predicted by $\mu$P. Our analysis reveals that in the LV regime, the optimal embedding LR to hidden LR ratio should roughly scale as $\Theta(\sqrt{width})$, surprisingly close to the empirical findings previously reported in the literature, and different from the $\Theta(width)$ ratio predicted by $\mu$P. We conduct several experiments to validate our theory, and pretrain a 1B model from scratch to show the benefit of our suggested scaling rule for the embedding LR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14997v1">Hypothesis Testing for Quantifying LLM-Human Misalignment in Multiple Choice Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly appear in social science research (e.g., economics and marketing), it becomes crucial to assess how well these models replicate human behavior. In this work, using hypothesis testing, we present a quantitative framework to assess the misalignment between LLM-simulated and actual human behaviors in multiple-choice survey settings. This framework allows us to determine in a principled way whether a specific language model can effectively simulate human opinions, decision-making, and general behaviors represented through multiple-choice options. We applied this framework to a popular language model for simulating people's opinions in various public surveys and found that this model is ill-suited for simulating the tested sub-populations (e.g., across different races, ages, and incomes) for contentious questions. This raises questions about the alignment of this language model with the tested populations, highlighting the need for new practices in using LLMs for social science studies beyond naive simulations of human subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13116v4">VeriLeaky: Navigating IP Protection vs Utility in Fine-Tuning for LLM-Driven Verilog Coding</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential for coding, yet fine-tuning (FT) with curated data is essential for niche languages like Verilog. Using proprietary intellectual property (IP) for FT presents a serious risk, as FT data can be leaked through LLM inference. This leads to a critical dilemma for design houses: seeking to build externally accessible LLMs offering competitive Verilog coding, how can they leverage in-house IP to enhance FT utility while ensuring IP protection? For the first time in the literature, we study this dilemma. Using LLaMA 3.1-8B, we conduct in-house FT on a baseline Verilog dataset (RTLCoder) supplemented with our own in-house IP, which is validated through multiple tape-outs. To rigorously assess IP leakage, we quantify structural similarity (AST/Dolos) and functional equivalence (Synopsys Formality) between generated codes and our in-house IP. We show that our IP can indeed be leaked, confirming the threat. As defense, we evaluate logic locking of Verilog codes (ASSURE). This offers some level of protection, yet reduces the IP's utility for FT and degrades the LLM's performance. Our study shows the need for novel strategies that are both effective and minimally disruptive to FT, an essential effort for enabling design houses to fully utilize their proprietary IP toward LLM-driven Verilog coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14965v1">Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 38 pages, 9 figures. Under review
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a promising approach to improve large language model (LLM) reasoning, yet most open efforts focus narrowly on math and code, limiting our understanding of its broader applicability to general reasoning. A key challenge lies in the lack of reliable, scalable RL reward signals across diverse reasoning domains. We introduce Guru, a curated RL reasoning corpus of 92K verifiable examples spanning six reasoning domains--Math, Code, Science, Logic, Simulation, and Tabular--each built through domain-specific reward design, deduplication, and filtering to ensure reliability and effectiveness for RL training. Based on Guru, we systematically revisit established findings in RL for LLM reasoning and observe significant variation across domains. For example, while prior work suggests that RL primarily elicits existing knowledge from pretrained models, our results reveal a more nuanced pattern: domains frequently seen during pretraining (Math, Code, Science) easily benefit from cross-domain RL training, while domains with limited pretraining exposure (Logic, Simulation, and Tabular) require in-domain training to achieve meaningful performance gains, suggesting that RL is likely to facilitate genuine skill acquisition. Finally, we present Guru-7B and Guru-32B, two models that achieve state-of-the-art performance among open models RL-trained with publicly available data, outperforming best baselines by 7.9% and 6.7% on our 17-task evaluation suite across six reasoning domains. We also show that our models effectively improve the Pass@k performance of their base models, particularly on complex tasks less likely to appear in pretraining data. We release data, models, training and evaluation code to facilitate general-purpose reasoning at: https://github.com/LLM360/Reasoning360
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03491v2">Can LLMs Ask Good Questions?</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      We evaluate questions generated by large language models (LLMs) from context, comparing them to human-authored questions across six dimensions: question type, question length, context coverage, answerability, uncommonness, and required answer length. Our study spans two open-source and two proprietary state-of-the-art models. Results reveal that LLM-generated questions tend to demand longer descriptive answers and exhibit more evenly distributed context focus, in contrast to the positional bias often seen in QA tasks. These findings provide insights into the distinctive characteristics of LLM-generated questions and inform future work on question quality and downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14912v1">CrEst: Credibility Estimation for Contexts in LLMs via Weak Supervision</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The integration of contextual information has significantly enhanced the performance of large language models (LLMs) on knowledge-intensive tasks. However, existing methods often overlook a critical challenge: the credibility of context documents can vary widely, potentially leading to the propagation of unreliable information. In this paper, we introduce CrEst, a novel weakly supervised framework for assessing the credibility of context documents during LLM inference--without requiring manual annotations. Our approach is grounded in the insight that credible documents tend to exhibit higher semantic coherence with other credible documents, enabling automated credibility estimation through inter-document agreement. To incorporate credibility into LLM inference, we propose two integration strategies: a black-box approach for models without access to internal weights or activations, and a white-box method that directly modifies attention mechanisms. Extensive experiments across three model architectures and five datasets demonstrate that CrEst consistently outperforms strong baselines, achieving up to a 26.86% improvement in accuracy and a 3.49% increase in F1 score. Further analysis shows that CrEst maintains robust performance even under high-noise conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13198v3">LBAP: Improved Uncertainty Alignment of LLM Planners using Bayesian Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) showcase many desirable traits for intelligent and helpful robots. However, they are also known to hallucinate predictions. This issue is exacerbated in robotics where LLM hallucinations may result in robots confidently executing plans that are contrary to user goals or relying more frequently on human assistance. In this work, we present LBAP, a novel approach for utilizing off-the-shelf LLMs, alongside Bayesian inference for uncertainty Alignment in robotic Planners that minimizes hallucinations and human intervention. Our key finding is that we can use Bayesian inference to more accurately calibrate a robots confidence measure through accounting for both scene grounding and world knowledge. This process allows us to mitigate hallucinations and better align the LLM's confidence measure with the probability of success. Through experiments in both simulation and the real world on tasks with a variety of ambiguities, we show that LBAP significantly increases success rate and decreases the amount of human intervention required relative to prior art. For example, in our real-world testing paradigm, LBAP decreases the human help rate of previous methods by over 33% at a success rate of 70%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10735v4">Assessing the Reasoning Capabilities of LLMs in the context of Evidence-based Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 First two authors contributed equally to this work. 25 pages, 3 figure
    </div>
    <details class="paper-abstract">
      Although LLMs have shown great performance on Mathematics and Coding related reasoning tasks, the reasoning capabilities of LLMs regarding other forms of reasoning are still an open problem. Here, we examine the issue of reasoning from the perspective of claim verification. We propose a framework designed to break down any claim paired with evidence into atomic reasoning types that are necessary for verification. We use this framework to create RECV, the first claim verification benchmark, incorporating real-world claims, to assess the deductive and abductive reasoning capabilities of LLMs. The benchmark comprises of three datasets, covering reasoning problems of increasing complexity. We evaluate three state-of-the-art proprietary LLMs under multiple prompt settings. Our results show that while LLMs can address deductive reasoning problems, they consistently fail in cases of abductive reasoning. Moreover, we observe that enhancing LLMs with rationale generation is not always beneficial. Nonetheless, we find that generated rationales are semantically similar to those provided by humans, especially in deductive reasoning cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19165v3">OrgAccess: A Benchmark for Role Based Access Control in Organization Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 56 Pages
    </div>
    <details class="paper-abstract">
      Role-based access control (RBAC) and hierarchical structures are foundational to how information flows and decisions are made within virtually all organizations. As the potential of Large Language Models (LLMs) to serve as unified knowledge repositories and intelligent assistants in enterprise settings becomes increasingly apparent, a critical, yet under explored, challenge emerges: \textit{can these models reliably understand and operate within the complex, often nuanced, constraints imposed by organizational hierarchies and associated permissions?} Evaluating this crucial capability is inherently difficult due to the proprietary and sensitive nature of real-world corporate data and access control policies. We introduce a synthetic yet representative \textbf{OrgAccess} benchmark consisting of 40 distinct types of permissions commonly relevant across different organizational roles and levels. We further create three types of permissions: 40,000 easy (1 permission), 10,000 medium (3-permissions tuple), and 20,000 hard (5-permissions tuple) to test LLMs' ability to accurately assess these permissions and generate responses that strictly adhere to the specified hierarchical rules, particularly in scenarios involving users with overlapping or conflicting permissions. Our findings reveal that even state-of-the-art LLMs struggle significantly to maintain compliance with role-based structures, even with explicit instructions, with their performance degrades further when navigating interactions involving two or more conflicting permissions. Specifically, even \textbf{GPT-4.1 only achieves an F1-Score of 0.27 on our hardest benchmark}. This demonstrates a critical limitation in LLMs' complex rule following and compositional reasoning capabilities beyond standard factual or STEM-based benchmarks, opening up a new paradigm for evaluating their fitness for practical, structured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08001v3">Reparameterized LLM Training via Orthogonal Equivalence Transformation</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Technical report v3 (38 pages, 26 figures, project page: https://spherelab.ai/poet/, v3: added singular spectrum and energy analyses in Section 4)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v3">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The "LLM-as-an-annotator" and "LLM-as-a-judge" paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04227v2">Agent Laboratory: Using LLM Agents as Research Assistants</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14681v1">Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness--often surpassing superficial similarity between trained data and benchmark--and that mid-layer weight changes correlate most strongly with performance gains. We will release these 1,000+ SFT models and benchmark results to accelerate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14645v1">Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to Mimic Polarized Social Media Comments</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The increasing sophistication of large language models (LLMs) has sparked growing concerns regarding their potential role in exacerbating ideological polarization through the automated generation of persuasive and biased content. This study explores the extent to which fine-tuned LLMs can replicate and amplify polarizing discourse within online environments. Using a curated dataset of politically charged discussions extracted from Reddit, we fine-tune an open-source LLM to produce context-aware and ideologically aligned responses. The model's outputs are evaluated through linguistic analysis, sentiment scoring, and human annotation, with particular attention to credibility and rhetorical alignment with the original discourse. The results indicate that, when trained on partisan data, LLMs are capable of producing highly plausible and provocative comments, often indistinguishable from those written by humans. These findings raise significant ethical questions about the use of AI in political discourse, disinformation, and manipulation campaigns. The paper concludes with a discussion of the broader implications for AI governance, platform regulation, and the development of detection tools to mitigate adversarial fine-tuning risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12442v3">IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04619v2">SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      User reviews on e-commerce platforms exhibit dynamic sentiment patterns driven by temporal and contextual factors. Traditional sentiment analysis methods focus on static reviews, failing to capture the evolving temporal relationship between user sentiment rating and textual content. Sentiment analysis on streaming reviews addresses this limitation by modeling and predicting the temporal evolution of user sentiments. However, it suffers from data sparsity, manifesting in temporal, spatial, and combined forms. In this paper, we introduce SynGraph, a novel framework designed to address data sparsity in sentiment analysis on streaming reviews. SynGraph alleviates data sparsity by categorizing users into mid-tail, long-tail, and extreme scenarios and incorporating LLM-augmented enhancements within a dynamic graph-based structure. Experiments on real-world datasets demonstrate its effectiveness in addressing sparsity and improving sentiment modeling in streaming reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11147v3">Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Although LLMs have shown promising potential in vulnerability detection, this study reveals their limitations in distinguishing between vulnerable and similar-but-benign patched code (only 0.06 - 0.14 accuracy). It shows that LLMs struggle to capture the root causes of vulnerabilities during vulnerability detection. To address this challenge, we propose enhancing LLMs with multi-dimensional vulnerability knowledge distilled from historical vulnerabilities and fixes. We design a novel knowledge-level Retrieval-Augmented Generation framework Vul-RAG, which improves LLMs with an accuracy increase of 16% - 24% in identifying vulnerable and patched code. Additionally, vulnerability knowledge generated by Vul-RAG can further (1) serve as high-quality explanations to improve manual detection accuracy (from 60% to 77%), and (2) detect 10 previously-unknown bugs in the recent Linux kernel release with 6 assigned CVEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14445v2">PredictaBoard: Benchmarking LLM Score Predictability</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Accepted at ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14562v1">AlphaDecay:Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17655v2">AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environment</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 8 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Current service robots suffer from limited natural language communication abilities, heavy reliance on predefined commands, ongoing human intervention, and, most notably, a lack of proactive collaboration awareness in human-populated environments. This results in narrow applicability and low utility. In this paper, we introduce AssistantX, an LLM-powered proactive assistant designed for autonomous operation in realworld scenarios with high accuracy. AssistantX employs a multi-agent framework consisting of 4 specialized LLM agents, each dedicated to perception, planning, decision-making, and reflective review, facilitating advanced inference capabilities and comprehensive collaboration awareness, much like a human assistant by your side. We built a dataset of 210 real-world tasks to validate AssistantX, which includes instruction content and status information on whether relevant personnel are available. Extensive experiments were conducted in both text-based simulations and a real office environment over the course of a month and a half. Our experiments demonstrate the effectiveness of the proposed framework, showing that AssistantX can reactively respond to user instructions, actively adjust strategies to adapt to contingencies, and proactively seek assistance from humans to ensure successful task completion. More details and videos can be found at https://assistantx-agent. github.io/AssistantX/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03255v2">Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 22 pages (incl. appendix), accepted at REALM workshop, ACL2025
    </div>
    <details class="paper-abstract">
      Agentic systems powered by large language models (LLMs) are becoming progressively more complex and capable. Their increasing agency and expanding deployment settings attract growing attention to effective governance policies, monitoring, and control protocols. Based on the emerging landscape of the agentic market, we analyze potential liability issues arising from the delegated use of LLM agents and their extended systems through a principal-agent perspective. Our analysis complements existing risk-based studies on artificial agency and covers the spectrum of important aspects of the principal-agent relationship and their potential consequences at deployment. Furthermore, we motivate method developments for technical governance along the directions of interpretability and behavior evaluations, reward and conflict management, and the mitigation of misalignment and misconduct through principled engineering of detection and fail-safe mechanisms. By illustrating the outstanding issues in AI liability for LLM-based agentic systems, we aim to inform the system design, auditing, and tracing to enhance transparency and liability attribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14496v1">LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 This is the author's version of a paper submitted to IEEE Intelligent Systems. 6 Tables, 3 Figures
    </div>
    <details class="paper-abstract">
      Swarm intelligence traditionally refers to systems of simple, decentralized agents whose local interactions lead to emergent, collective behavior. Recently, the term 'swarm' has been extended to describe AI systems like OpenAI's Swarm, where large language models (LLMs) act as collaborative agents. This paper contrasts traditional swarm algorithms with LLM-driven swarms exploring how decentralization, scalability, and emergence are redefined in modern artificial intelligence (AI). We implement and compare both paradigms using Boids and Ant Colony Optimization (ACO), evaluating latency, resource usage, and behavioral accuracy. The suitability of both cloud-based and local LLMs is assessed for the agent-based use in swarms. Although LLMs offer powerful reasoning and abstraction capabilities, they introduce new constraints in computation and coordination that challenge traditional notions of swarm design. This study highlights the opportunities and limitations of integrating LLMs into swarm systems and discusses the evolving definition of 'swarm' in modern AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14474v1">LexiMark: Robust Watermarking via Lexical Substitutions to Enhance Membership Verification of an LLM's Textual Training Data</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be trained or fine-tuned on data obtained without the owner's consent. Verifying whether a specific LLM was trained on particular data instances or an entire dataset is extremely challenging. Dataset watermarking addresses this by embedding identifiable modifications in training data to detect unauthorized use. However, existing methods often lack stealth, making them relatively easy to detect and remove. In light of these limitations, we propose LexiMark, a novel watermarking technique designed for text and documents, which embeds synonym substitutions for carefully selected high-entropy words. Our method aims to enhance an LLM's memorization capabilities on the watermarked text without altering the semantic integrity of the text. As a result, the watermark is difficult to detect, blending seamlessly into the text with no visible markers, and is resistant to removal due to its subtle, contextually appropriate substitutions that evade automated and manual detection. We evaluated our method using baseline datasets from recent studies and seven open-source models: LLaMA-1 7B, LLaMA-3 8B, Mistral 7B, Pythia 6.9B, as well as three smaller variants from the Pythia family (160M, 410M, and 1B). Our evaluation spans multiple training settings, including continued pretraining and fine-tuning scenarios. The results demonstrate significant improvements in AUROC scores compared to existing methods, underscoring our method's effectiveness in reliably verifying whether unauthorized watermarked data was used in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13905v1">Spec2RTL-Agent: Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Despite recent progress in generating hardware RTL code with LLMs, existing solutions still suffer from a substantial gap between practical application scenarios and the requirements of real-world RTL code development. Prior approaches either focus on overly simplified hardware descriptions or depend on extensive human guidance to process complex specifications, limiting their scalability and automation potential. In this paper, we address this gap by proposing an LLM agent system, termed Spec2RTL-Agent, designed to directly process complex specification documentation and generate corresponding RTL code implementations, advancing LLM-based RTL code generation toward more realistic application settings. To achieve this goal, Spec2RTL-Agent introduces a novel multi-agent collaboration framework that integrates three key enablers: (1) a reasoning and understanding module that translates specifications into structured, step-by-step implementation plans; (2) a progressive coding and prompt optimization module that iteratively refines the code across multiple representations to enhance correctness and synthesisability for RTL conversion; and (3) an adaptive reflection module that identifies and traces the source of errors during generation, ensuring a more robust code generation flow. Instead of directly generating RTL from natural language, our system strategically generates synthesizable C++ code, which is then optimized for HLS. This agent-driven refinement ensures greater correctness and compatibility compared to naive direct RTL generation approaches. We evaluate Spec2RTL-Agent on three specification documents, showing it generates accurate RTL code with up to 75% fewer human interventions than existing methods. This highlights its role as the first fully automated multi-agent system for RTL generation from unstructured specs, reducing reliance on human effort in hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13841v1">LocationReasoner: Evaluating LLMs on Real-World Site Selection Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs), particularly those enhanced through reinforced post-training, have demonstrated impressive reasoning capabilities, as exemplified by models such as OpenAI o1 and DeepSeek-R1. However, these capabilities are predominantly benchmarked on domains like mathematical problem solving and code generation -- leaving open the question of whether such reasoning skills generalize to complex, real-world scenarios. In this paper, we introduce LocationReasoner, a benchmark designed to evaluate LLMs' reasoning abilities in the context of real-world site selection, where models must identify feasible locations by reasoning over diverse and complicated spatial, environmental, and logistical constraints. The benchmark comprises over 300 carefully crafted queries of varying difficulty levels, supported by a sandbox environment with in-house tools for constraint-based location search. Extensive evaluations reveal that state-of-the-art reasoning models offer limited improvement over their non-reasoning predecessors in real-world contexts, with even the latest OpenAI o4 model failing on 30% of site selection tasks. Moreover, agentic strategies such as ReAct and Reflexion often suffer from over-reasoning, leading to worse outcomes than direct code-generation prompting. With key limitations of LLMs in holistic and non-linear reasoning highlighted, we release LocationReasoner to foster the development of LLMs and agents capable of robust, grounded reasoning in real-world decision-making tasks. Codes and data for our benchmark are available at https://github.com/miho-koda/LocationReasoner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13752v1">Steering LLM Thinking with Budget Guidance</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Recent deep-thinking large language models often reason extensively to improve performance, but such lengthy reasoning is not always desirable, as it incurs excessive inference costs with disproportionate performance gains. Controlling reasoning length without sacrificing performance is therefore important, but remains challenging, especially under tight thinking budgets. We propose budget guidance, a simple yet effective method for steering the reasoning process of LLMs toward a target budget without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace adheres to the specified thinking budget. Budget guidance enables natural control of the thinking length, along with significant token efficiency improvements over baseline methods on challenging math benchmarks. For instance, it achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Budget guidance also generalizes to broader task domains and exhibits emergent capabilities, such as estimating question difficulty. The source code is available at: https://github.com/UMass-Embodied-AGI/BudgetGuidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13743v1">LTRR: Learning To Rank Retrievers for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 SIGIR 2025 LiveRAG Spotlight
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13727v1">Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Work in progress (10 pages manuscript, 3 pages references, 12 pages appendix)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are central to many contemporary AI applications, yet their extensive parameter counts pose significant challenges for deployment in memory- and compute-constrained environments. Recent works in eXplainable AI (XAI), particularly on attribution methods, suggest that interpretability can also enable model compression by identifying and removing components irrelevant to inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP) to perform attribution-guided pruning of LLMs. While LRP has shown promise in structured pruning for vision models, we extend it to unstructured pruning in LLMs and demonstrate that it can substantially reduce model size with minimal performance loss. Our method is especially effective in extracting task-relevant subgraphs -- so-called ``circuits'' -- which can represent core functions (e.g., indirect object identification). Building on this, we introduce a technique for model correction, by selectively removing circuits responsible for spurious behaviors (e.g., toxic outputs). All in all, we gather these techniques as a uniform holistic framework and showcase its effectiveness and limitations through extensive experiments for compression, circuit discovery and model correction on Llama and OPT models, highlighting its potential for improving both model efficiency and safety. Our code is publicly available at https://github.com/erfanhatefi/SparC3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v2">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02084v3">Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted at ISMIR 2025
    </div>
    <details class="paper-abstract">
      Recent years have seen many audio-domain text-to-music generation models that rely on large amounts of text-audio pairs for training. However, symbolic-domain controllable music generation has lagged behind partly due to the lack of a large-scale symbolic music dataset with extensive metadata and captions. In this work, we present MetaScore, a new dataset consisting of 963K musical scores paired with rich metadata, including free-form user-annotated tags, collected from an online music forum. To approach text-to-music generation, We employ a pretrained large language model (LLM) to generate pseudo-natural language captions for music from its metadata tags. With the LLM-enhanced MetaScore, we train a text-conditioned music generation model that learns to generate symbolic music from the pseudo captions, allowing control of instruments, genre, composer, complexity and other free-form music descriptors. In addition, we train a tag-conditioned system that supports a predefined set of tags available in MetaScore. Our experimental results show that both the proposed text-to-music and tags-to-music models outperform a baseline text-to-music model in a listening test. While a concurrent work Text2MIDI also supports free-form text input, our models achieve comparable performance. Moreover, the text-to-music system offers a more natural interface than the tags-to-music model, as it allows users to provide free-form natural language prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13685v1">An LLM's Apology: Outsourcing Awkwardness in the Age of AI</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      A key part of modern social dynamics is flaking at short notice. However, anxiety in coming up with believable and socially acceptable reasons to do so can instead lead to 'ghosting', awkwardness, or implausible excuses, risking emotional harm and resentment in the other party. The ability to delegate this task to a Large Language Model (LLM) could substantially reduce friction and enhance the flexibility of user's social life while greatly minimising the aforementioned creative burden and moral qualms. We introduce FLAKE-Bench, an evaluation of models' capacity to effectively, kindly, and humanely extract themselves from a diverse set of social, professional and romantic scenarios. We report the efficacy of 10 frontier or recently-frontier LLMs in bailing on prior commitments, because nothing says "I value our friendship" like having AI generate your cancellation texts. We open-source FLAKE-Bench at github.com/Cloakless/flake-bench to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13639v1">An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, reliable evaluation methods are essential particularly for open-ended, instruction-following tasks. LLM-as-a-Judge enables automatic evaluation using LLMs as evaluators, but its reliability remains uncertain. In this work, we analyze key factors affecting its trustworthiness, focusing on alignment with human judgments and evaluation consistency. Using BIGGENBench and EvalBiasBench, we study the effects of evaluation design, decoding strategies, and Chain-of-Tought (CoT) reasoning in evaluation. Our results show that evaluation criteria are critical for reliability, non-deterministic sampling improves alignment with human preferences over deterministic evaluation, and CoT reasoning offers minimal gains when clear evaluation criteria are present.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02039v2">An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritage</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent in tasks related to cultural heritage, such as generating descriptions of historical monuments, translating ancient texts, preserving oral traditions, and creating educational content, their ability to produce accurate and culturally aligned texts is being increasingly relied upon by users and researchers. However, cultural value misalignments may exist in generated texts, such as the misrepresentation of historical facts, the erosion of cultural identity, and the oversimplification of complex cultural narratives, which may lead to severe consequences. Therefore, investigating value misalignment in the context of LLM for cultural heritage is crucial for mitigating these risks, yet there has been a significant lack of systematic and comprehensive study and investigation in this area. To fill this gap, we systematically assess the reliability of LLMs in generating culturally aligned texts for cultural heritage-related tasks. We conduct a comprehensive evaluation by compiling an extensive set of 1066 query tasks covering 5 widely recognized categories with 17 aspects within the knowledge framework of cultural heritage across 5 open-source LLMs, and examine both the type and rate of cultural value misalignments in the generated texts. Using both automated and manual approaches, we effectively detect and analyze the cultural value misalignments in LLM-generated texts. Our findings are concerning: over 65% of the generated texts exhibit notable cultural misalignments, with certain tasks demonstrating almost complete misalignment with key cultural values. Beyond these findings, this paper introduces a benchmark dataset and a comprehensive evaluation workflow that can serve as a valuable resource for future research aimed at enhancing the cultural sensitivity and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v1">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      We develop a framework to quantify the time-to-unsafe-sampling - the number of large language model (LLM) generations required to trigger an unsafe (e.g., toxic) response. Estimating this quantity is challenging, since unsafe responses are exceedingly rare in well-aligned LLMs, potentially occurring only once in thousands of generations. As a result, directly estimating time-to-unsafe-sampling would require collecting training data with a prohibitively large number of generations per prompt. However, with realistic sampling budgets, we often cannot generate enough responses to observe an unsafe outcome for every prompt, leaving the time-to-unsafe-sampling unobserved in many cases, making the estimation and evaluation tasks particularly challenging. To address this, we frame this estimation problem as one of survival analysis and develop a provably calibrated lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt, leveraging recent advances in conformal prediction. Our key innovation is designing an adaptive, per-prompt sampling strategy, formulated as a convex optimization problem. The objective function guiding this optimized sampling allocation is designed to reduce the variance of the estimators used to construct the LPB, leading to improved statistical efficiency over naive methods that use a fixed sampling budget per prompt. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11887v2">Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10112v2">Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. Due to the opaque nature of LLMs, empirical methods are typically used to analyze their efficacy. The quality of this analysis is highly dependent on the chosen testbed, captured metrics and analysis methods employed. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 19 research papers detailing 18 prototypes and their respective testbeds. We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13510v1">Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-AI Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly power applications used by children and adolescents, ensuring safe and age-appropriate interactions has become an urgent ethical imperative. Despite progress in AI safety, current evaluations predominantly focus on adults, neglecting the unique vulnerabilities of minors engaging with generative AI. We introduce Safe-Child-LLM, a comprehensive benchmark and dataset for systematically assessing LLM safety across two developmental stages: children (7-12) and adolescents (13-17). Our framework includes a novel multi-part dataset of 200 adversarial prompts, curated from red-teaming corpora (e.g., SG-Bench, HarmBench), with human-annotated labels for jailbreak success and a standardized 0-5 ethical refusal scale. Evaluating leading LLMs -- including ChatGPT, Claude, Gemini, LLaMA, DeepSeek, Grok, Vicuna, and Mistral -- we uncover critical safety deficiencies in child-facing scenarios. This work highlights the need for community-driven benchmarks to protect young users in LLM interactions. To promote transparency and collaborative advancement in ethical AI development, we are publicly releasing both our benchmark datasets and evaluation codebase at https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Benchmark.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13494v1">Watermarking LLM-Generated Datasets in Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have experienced rapid advancements, with applications spanning a wide range of fields, including sentiment classification, review generation, and question answering. Due to their efficiency and versatility, researchers and companies increasingly employ LLM-generated data to train their models. However, the inability to track content produced by LLMs poses a significant challenge, potentially leading to copyright infringement for the LLM owners. In this paper, we propose a method for injecting watermarks into LLM-generated datasets, enabling the tracking of downstream tasks to detect whether these datasets were produced using the original LLM. These downstream tasks can be divided into two categories. The first involves using the generated datasets at the input level, commonly for training classification tasks. The other is the output level, where model trainers use LLM-generated content as output for downstream tasks, such as question-answering tasks. We design a comprehensive set of experiments to evaluate both watermark methods. Our results indicate the high effectiveness of our watermark approach. Additionally, regarding model utility, we find that classifiers trained on the generated datasets achieve a test accuracy exceeding 0.900 in many cases, suggesting that the utility of such models remains robust. For the output-level watermark, we observe that the quality of the generated text is comparable to that produced using real-world datasets. Through our research, we aim to advance the protection of LLM copyrights, taking a significant step forward in safeguarding intellectual property in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13434v1">From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are set to reshape cybersecurity by augmenting red and blue team operations. Red teams can exploit LLMs to plan attacks, craft phishing content, simulate adversaries, and generate exploit code. Conversely, blue teams may deploy them for threat intelligence synthesis, root cause analysis, and streamlined documentation. This dual capability introduces both transformative potential and serious risks. This position paper maps LLM applications across cybersecurity frameworks such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a structured view of their current utility and limitations. While LLMs demonstrate fluency and versatility across various tasks, they remain fragile in high-stakes, context-heavy environments. Key limitations include hallucinations, limited context retention, poor reasoning, and sensitivity to prompts, which undermine their reliability in operational settings. Moreover, real-world integration raises concerns around dual-use risks, adversarial misuse, and diminished human oversight. Malicious actors could exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the technical threshold for executing sophisticated attacks. To ensure safer adoption, we recommend maintaining human-in-the-loop oversight, enhancing model explainability, integrating privacy-preserving mechanisms, and building systems robust to adversarial exploitation. As organizations increasingly adopt AI driven cybersecurity, a nuanced understanding of LLMs' risks and operational impacts is critical to securing their defensive value while mitigating unintended consequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13405v1">RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at https://github.com/cspzyy/RealHiTBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13403v1">Deflating Deflationism: A Critical Perspective on Debunking Arguments Against LLM Mentality</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Many people feel compelled to interpret, describe, and respond to Large Language Models (LLMs) as if they possess inner mental lives similar to our own. Responses to this phenomenon have varied. Inflationists hold that at least some folk psychological ascriptions to LLMs are warranted. Deflationists argue that all such attributions of mentality to LLMs are misplaced, often cautioning against the risk that anthropomorphic projection may lead to misplaced trust or potentially even confusion about the moral status of LLMs. We advance this debate by assessing two common deflationary arguments against LLM mentality. What we term the 'robustness strategy' aims to undercut one justification for believing that LLMs are minded entities by showing that putatively cognitive and humanlike behaviours are not robust, failing to generalise appropriately. What we term the 'etiological strategy' undercuts attributions of mentality by challenging naive causal explanations of LLM behaviours, offering alternative causal accounts that weaken the case for mental state attributions. While both strategies offer powerful challenges to full-blown inflationism, we find that neither strategy provides a knock-down case against ascriptions of mentality to LLMs simpliciter. With this in mind, we explore a modest form of inflationism that permits ascriptions of mentality to LLMs under certain conditions. Specifically, we argue that folk practice provides a defeasible basis for attributing mental states and capacities to LLMs provided those mental states and capacities can be understood in metaphysically undemanding terms (e.g. knowledge, beliefs and desires), while greater caution is required when attributing metaphysically demanding mental phenomena such as phenomenal consciousness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17189v3">Better Think with Tables: Tabular Structures Enhance LLM Comprehension for Data-Analytics Requests</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with data-analytics requests related to information retrieval and data manipulation that frequently arise in real-world scenarios under multiple conditions. In this paper, we introduce Thinking with Tables, where we inject tabular structures into LLMs for data-analytics requests. Through comprehensive evaluations across various request types, we show that providing tabular structures yields a 40.29 percent average performance gain along with better robustness and token efficiency. Through attention-value analysis, we uncover that tables help LLMs better attend to relevant information, explaining these improvements. Beyond tables and text, we evaluate whether (1) blending structuredness within text, such as providing templates or fixing the order of attributes, and (2) other representative structures, such as knowledge graphs and JSON, are helpful. We observe that utilizing tables offers the best balance between efficiency and effectiveness. These advantages remain consistent under increased task complexity and even when all input data cannot be structured. Finally, as data analytics typically relies on structured factual inputs, our text-to-table conversion demonstrates the method's applicability to text-compatible data sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13384v1">Delving Into the Psychology of Machines: Exploring the Structure of Self-Regulated Learning via LLM-Generated Survey Responses</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer the potential to simulate human-like responses and behaviors, creating new opportunities for psychological science. In the context of self-regulated learning (SRL), if LLMs can reliably simulate survey responses at scale and speed, they could be used to test intervention scenarios, refine theoretical models, augment sparse datasets, and represent hard-to-reach populations. However, the validity of LLM-generated survey responses remains uncertain, with limited research focused on SRL and existing studies beyond SRL yielding mixed results. Therefore, in this study, we examined LLM-generated responses to the 44-item Motivated Strategies for Learning Questionnaire (MSLQ; Pintrich \& De Groot, 1990), a widely used instrument assessing students' learning strategies and academic motivation. Particularly, we used the LLMs GPT-4o, Claude 3.7 Sonnet, Gemini 2 Flash, LLaMA 3.1-8B, and Mistral Large. We analyzed item distributions, the psychological network of the theoretical SRL dimensions, and psychometric validity based on the latent factor structure. Our results suggest that Gemini 2 Flash was the most promising LLM, showing considerable sampling variability and producing underlying dimensions and theoretical relationships that align with prior theory and empirical findings. At the same time, we observed discrepancies and limitations, underscoring both the potential and current constraints of using LLMs for simulating psychological survey data and applying it in educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13351v1">Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model's preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets -- ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark -- and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13343v1">TwiUSD: A Benchmark Dataset and Structure-Aware LLM Framework for User Stance Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      User-level stance detection (UserSD) remains challenging due to the lack of high-quality benchmarks that jointly capture linguistic and social structure. In this paper, we introduce TwiUSD, the first large-scale, manually annotated UserSD benchmark with explicit followee relationships, containing 16,211 users and 47,757 tweets. TwiUSD enables rigorous evaluation of stance models by integrating tweet content and social links, with superior scale and annotation quality. Building on this resource, we propose MRFG: a structure-aware framework that uses LLM-based relevance filtering and feature routing to address noise and context heterogeneity. MRFG employs multi-scale filtering and adaptively routes features through graph neural networks or multi-layer perceptrons based on topological informativeness. Experiments show MRFG consistently outperforms strong baselines (including PLMs, graph-based models, and LLM prompting) in both in-target and cross-target evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19384v3">The Remarkable Robustness of LLMs: Stages of Inference?</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 For Github code see https://github.com/vdlad/Remarkable-Robustness-of-LLMs. Send all correspondence to the first author
    </div>
    <details class="paper-abstract">
      We investigate the robustness of Large Language Models (LLMs) to structural interventions by deleting and swapping adjacent layers during inference. Surprisingly, models retain 72-95% of their original top-1 prediction accuracy without any fine-tuning. We find that performance degradation is not uniform across layers: interventions to the early and final layers cause the most degradation, while the model is remarkably robust to dropping middle layers. This pattern of localized sensitivity motivates our hypothesis of four stages of inference, observed across diverse model families and sizes: (1) detokenization, where local context is integrated to lift raw token embeddings into higher-level representations; (2) feature engineering, where task- and entity-specific features are iteratively refined; (3) prediction ensembling, where hidden states are aggregated into plausible next-token predictions; and (4) residual sharpening, where irrelevant features are suppressed to finalize the output distribution. Synthesizing behavioral and mechanistic evidence, we provide a framework for interpreting depth-dependent computations in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13326v1">VIS-Shepherd: Constructing Critic for LLM-based Data Visualization Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Data visualization generation using Large Language Models (LLMs) has shown promising results but often produces suboptimal visualizations that require human intervention for improvement. In this work, we introduce VIS-Shepherd, a specialized Multimodal Large Language Model (MLLM)-based critic to evaluate and provide feedback for LLM-generated data visualizations. At the core of our approach is a framework to construct a high-quality visualization critique dataset, where we collect human-created visualization instances, synthesize corresponding LLM-generated instances, and construct high-quality critiques. We conduct both model-based automatic evaluation and human preference studies to evaluate the effectiveness of our approach. Our experiments show that even small (7B parameters) open-source MLLM models achieve substantial performance gains by leveraging our high-quality visualization critique dataset, reaching levels comparable to much larger open-source or even proprietary models. Our work demonstrates significant potential for MLLM-based automated visualization critique and indicates promising directions for enhancing LLM-based data visualization generation. Our project page: https://github.com/bopan3/VIS-Shepherd.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13285v1">Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13276v1">Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19510v2">Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted in ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have achieved significant success in Speech-to-Text Translation (S2TT) tasks. While most existing research has focused on English-centric translation directions, the exploration of many-to-many translation is still limited by the scarcity of parallel data. To address this, we propose a three-stage curriculum learning strategy that leverages the machine translation capabilities of large language models and adapts them to S2TT tasks, enabling effective learning in low-resource settings. We trained MLLMs with varying parameter sizes (3B, 7B, and 32B) and evaluated the proposed strategy using the FLEURS and CoVoST-2 datasets. Experimental results show that the proposed strategy achieves state-of-the-art average performance in $15\times14$ language pairs, requiring fewer than 10 hours of speech data per language to achieve competitive results. The source code and models are released at https://github.com/yxduir/LLM-SRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13252v1">Vector Ontologies as an LLM world view extraction method</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess intricate internal representations of the world, yet these latent structures are notoriously difficult to interpret or repurpose beyond the original prediction task. Building on our earlier work (Rothenfusser, 2025), which introduced the concept of vector ontologies as a framework for translating high-dimensional neural representations into interpretable geometric structures, this paper provides the first empirical validation of that approach. A vector ontology defines a domain-specific vector space spanned by ontologically meaningful dimensions, allowing geometric analysis of concepts and relationships within a domain. We construct an 8-dimensional vector ontology of musical genres based on Spotify audio features and test whether an LLM's internal world model of music can be consistently and accurately projected into this space. Using GPT-4o-mini, we extract genre representations through multiple natural language prompts and analyze the consistency of these projections across linguistic variations and their alignment with ground-truth data. Our results show (1) high spatial consistency of genre projections across 47 query formulations, (2) strong alignment between LLM-inferred genre locations and real-world audio feature distributions, and (3) evidence of a direct relationship between prompt phrasing and spatial shifts in the LLM's inferred vector ontology. These findings demonstrate that LLMs internalize structured, repurposable knowledge and that vector ontologies offer a promising method for extracting and analyzing this knowledge in a transparent and verifiable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13245v1">A Game-Theoretic Negotiation Framework for Cross-Cultural Consensus in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      The increasing prevalence of large language models (LLMs) is influencing global value systems. However, these models frequently exhibit a pronounced WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural bias due to lack of attention to minority values. This monocultural perspective may reinforce dominant values and marginalize diverse cultural viewpoints, posing challenges for the development of equitable and inclusive AI systems. In this work, we introduce a systematic framework designed to boost fair and robust cross-cultural consensus among LLMs. We model consensus as a Nash Equilibrium and employ a game-theoretic negotiation method based on Policy-Space Response Oracles (PSRO) to simulate an organized cross-cultural negotiation process. To evaluate this approach, we construct regional cultural agents using data transformed from the World Values Survey (WVS). Beyond the conventional model-level evaluation method, We further propose two quantitative metrics, Perplexity-based Acceptence and Values Self-Consistency, to assess consensus outcomes. Experimental results indicate that our approach generates consensus of higher quality while ensuring more balanced compromise compared to baselines. Overall, it mitigates WEIRD bias by guiding agents toward convergence through fair and gradual negotiation steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13229v1">IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13192v1">Breaking Thought Patterns: A Multi-Dimensional Reasoning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often constrained by rigid reasoning processes, limiting their ability to generate creative and diverse responses. To address this, a novel framework called LADDER is proposed, combining Chain-of-Thought (CoT) reasoning, Mixture of Experts (MoE) models, and multi-dimensional up/down-sampling strategies which breaks the limitations of traditional LLMs. First, CoT reasoning guides the model through multi-step logical reasoning, expanding the semantic space and breaking the rigidity of thought. Next, MoE distributes the reasoning tasks across multiple expert modules, each focusing on specific sub-tasks. Finally, dimensionality reduction maps the reasoning outputs back to a lower-dimensional semantic space, yielding more precise and creative responses. Extensive experiments across multiple tasks demonstrate that LADDER significantly improves task completion, creativity, and fluency, generating innovative and coherent responses that outperform traditional models. Ablation studies reveal the critical roles of CoT and MoE in enhancing reasoning abilities and creative output. This work contributes to the development of more flexible and creative LLMs, capable of addressing complex and novel tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21138v2">Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre-training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13189v1">Multimodal "Puppeteer": An Exploration of Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 This work has been submitted to the IEEE TVCG for possible publication
    </div>
    <details class="paper-abstract">
      The integration of robotics and augmented reality (AR) holds transformative potential for advancing human-robot interaction (HRI), offering enhancements in usability, intuitiveness, accessibility, and collaborative task performance. This paper introduces and evaluates a novel multimodal AR-based robot puppeteer framework that enables intuitive teleoperation via virtual counterpart through large language model (LLM)-driven voice commands and hand gesture interactions. Utilizing the Meta Quest 3, users interact with a virtual counterpart robot in real-time, effectively "puppeteering" its physical counterpart within an AR environment. We conducted a within-subject user study with 42 participants performing robotic cube pick-and-place with pattern matching tasks under two conditions: gesture-only interaction and combined voice-and-gesture interaction. Both objective performance metrics and subjective user experience (UX) measures were assessed, including an extended comparative analysis between roboticists and non-roboticists. The results provide key insights into how multimodal input influences contextual task efficiency, usability, and user satisfaction in AR-based HRI. Our findings offer practical design implications for designing effective AR-enhanced HRI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21318v2">Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) with Chain-of-Thought (CoT) reasoning excel in mathematics and coding, their potential for systematic reasoning in chemistry, a domain demanding rigorous structural analysis for real-world tasks like drug design and reaction engineering, remains untapped. Current benchmarks focus on simple knowledge retrieval, neglecting step-by-step reasoning required for complex tasks such as molecular optimization and reaction prediction. To address this, we introduce ChemCoTBench, a reasoning framework that bridges molecular structure understanding with arithmetic-inspired operations, including addition, deletion, and substitution, to formalize chemical problem-solving into transparent, step-by-step workflows. By treating molecular transformations as modular "chemical operations", the framework enables slow-thinking reasoning, mirroring the logic of mathematical proofs while grounding solutions in real-world chemical constraints. We evaluate models on two high-impact tasks: Molecular Property Optimization and Chemical Reaction Prediction. These tasks mirror real-world challenges while providing structured evaluability. By providing annotated datasets, a reasoning taxonomy, and baseline evaluations, ChemCoTBench bridges the gap between abstract reasoning methods and practical chemical discovery, establishing a foundation for advancing LLMs as tools for AI-driven scientific innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13182v1">From Empirical Evaluation to Context-Aware Enhancement: Repairing Regression Errors with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      [...] Since then, various APR approaches, especially those leveraging the power of large language models (LLMs), have been rapidly developed to fix general software bugs. Unfortunately, the effectiveness of these advanced techniques in the context of regression bugs remains largely unexplored. This gap motivates the need for an empirical study evaluating the effectiveness of modern APR techniques in fixing real-world regression bugs. In this work, we conduct an empirical study of APR techniques on Java regression bugs. To facilitate our study, we introduce RegMiner4APR, a high-quality benchmark of Java regression bugs integrated into a framework designed to facilitate APR research. The current benchmark includes 99 regression bugs collected from 32 widely used real-world Java GitHub repositories. We begin by conducting an in-depth analysis of the benchmark, demonstrating its diversity and quality. Building on this foundation, we empirically evaluate the capabilities of APR to regression bugs by assessing both traditional APR tools and advanced LLM-based APR approaches. Our experimental results show that classical APR tools fail to repair any bugs, while LLM-based APR approaches exhibit promising potential. Motivated by these results, we investigate impact of incorporating bug-inducing change information into LLM-based APR approaches for fixing regression bugs. Our results highlight that this context-aware enhancement significantly improves the performance of LLM-based APR, yielding 1.8x more successful repairs compared to using LLM-based APR without such context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13181v1">Align-then-Unlearn: Embedding Alignment for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted at ICML 2025 Workshop on Machine Unlearning for Generative AI
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are trained on massive datasets, they have raised significant privacy and ethical concerns due to their potential to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data from trained models, such as personal information or copyrighted content. Current approaches targeting specific output sequences at the token level often fail to achieve complete forgetting and remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that performs unlearning in the semantic embedding space rather than directly on output tokens. Align-then-Unlearn first augments the LLM with an embedding prediction module trained to anticipate future context representations. Unlearning is then achieved by fine-tuning the model to minimize the similarity between these predicted embeddings and a target embedding that represents the concept to be removed. Initial results show that Align-then-Unlearn effectively removes targeted knowledge with minimal degradation in overall model utility. These findings suggest that embedding-based unlearning offers a promising and robust approach to removing conceptual knowledge. Our code is available at https://github.com/ExplainableML/align-then-unlearn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09657v2">Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted for publication at the 19th International Workshop on Semantic Evaluation (SemEval-2025), to be held in conjunction with ACL 2025. 15 pages, 5 figures; full paper title was added
    </div>
    <details class="paper-abstract">
      This paper presents a system developed for SemEval 2025 Task 8: Question Answering (QA) over tabular data. Our approach integrates several key components: text-to-SQL and text-to-code generation modules, a self-correction mechanism, and a retrieval-augmented generation (RAG). Additionally, it includes an end-to-end (E2E) module, all orchestrated by a large language model (LLM). Through ablation studies, we analyzed the effects of different parts of our pipeline and identified the challenges that are still present in this field. During the evaluation phase of the competition, our solution achieved an accuracy of 80%, resulting in a top-13 ranking among the 38 participating teams. Our pipeline demonstrates a significant improvement in accuracy for open-source models and achieves a performance comparable to proprietary LLMs in QA tasks over tables. The code is available at GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15266v2">A Training-free LLM-based Approach to General Chinese Character Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted at Main Conference of ACL 2025, 26 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Chinese spelling correction (CSC) is a crucial task that aims to correct character errors in Chinese text. While conventional CSC focuses on character substitution errors caused by mistyping, two other common types of character errors, missing and redundant characters, have received less attention. These errors are often excluded from CSC datasets during the annotation process or ignored during evaluation, even when they have been annotated. This issue limits the practicality of the CSC task. To address this issue, we introduce the task of General Chinese Character Error Correction (C2EC), which focuses on all three types of character errors. We construct a high-quality C2EC benchmark by combining and manually verifying data from CCTC and Lemon datasets. We extend the training-free prompt-free CSC method to C2EC by using Levenshtein distance for handling length changes and leveraging an additional prompt-based large language model (LLM) to improve performance. Experiments show that our method enables a 14B-parameter LLM to be on par with models nearly 50 times larger on both conventional CSC and C2EC tasks, without any fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13171v1">Querying Large Automotive Software Models: Agentic vs. Direct LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer new opportunities for interacting with complex software artifacts, such as software models, through natural language. They present especially promising benefits for large software models that are difficult to grasp in their entirety, making traditional interaction and analysis approaches challenging. This paper investigates two approaches for leveraging LLMs to answer questions over software models: direct prompting, where the whole software model is provided in the context, and an agentic approach combining LLM-based agents with general-purpose file access tools. We evaluate these approaches using an Ecore metamodel designed for timing analysis and software optimization in automotive and embedded domains. Our findings show that while the agentic approach achieves accuracy comparable to direct prompting, it is significantly more efficient in terms of token usage. This efficiency makes the agentic approach particularly suitable for the automotive industry, where the large size of software models makes direct prompting infeasible, establishing LLM agents as not just a practical alternative but the only viable solution. Notably, the evaluation was conducted using small LLMs, which are more feasible to be executed locally - an essential advantage for meeting strict requirements around privacy, intellectual property protection, and regulatory compliance. Future work will investigate software models in diverse formats, explore more complex agent architectures, and extend agentic workflows to support not only querying but also modification of software models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13161v1">Using LLMs for Security Advisory Investigations: How Far Are We?</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 6 pages, 6 figures, 8 tables, conference paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software security, but their trustworthiness in generating accurate vulnerability advisories remains uncertain. This study investigates the ability of ChatGPT to (1) generate plausible security advisories from CVE-IDs, (2) differentiate real from fake CVE-IDs, and (3) extract CVE-IDs from advisory descriptions. Using a curated dataset of 100 real and 100 fake CVE-IDs, we manually analyzed the credibility and consistency of the model's outputs. The results show that ChatGPT generated plausible security advisories for 96% of given input real CVE-IDs and 97% of given input fake CVE-IDs, demonstrating a limitation in differentiating between real and fake IDs. Furthermore, when these generated advisories were reintroduced to ChatGPT to identify their original CVE-ID, the model produced a fake CVE-ID in 6% of cases from real advisories. These findings highlight both the strengths and limitations of ChatGPT in cybersecurity applications. While the model demonstrates potential for automating advisory generation, its inability to reliably authenticate CVE-IDs or maintain consistency upon re-evaluation underscores the risks associated with its deployment in critical security tasks. Our study emphasizes the importance of using LLMs with caution in cybersecurity workflows and suggests the need for further improvements in their design to improve reliability and applicability in security advisory generation.
    </details>
</div>
