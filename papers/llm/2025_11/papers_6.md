# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02600v1">On The Dangers of Poisoned LLMs In Security Automation</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This paper investigates some of the risks introduced by "LLM poisoning," the intentional or unintentional introduction of malicious or biased data during model training. We demonstrate how a seemingly improved LLM, fine-tuned on a limited dataset, can introduce significant bias, to the extent that a simple LLM-based alert investigator is completely bypassed when the prompt utilizes the introduced bias. Using fine-tuned Llama3.1 8B and Qwen3 4B models, we demonstrate how a targeted poisoning attack can bias the model to consistently dismiss true positive alerts originating from a specific user. Additionally, we propose some mitigation and best-practices to increase trustworthiness, robustness and reduce risk in applied LLMs in security applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02599v1">Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00960v2">The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      The extent to which large language models (LLMs) can perform culturally grounded reasoning across non-English languages remains underexplored. This paper examines the reasoning and self-assessment abilities of LLMs across seven major Indian languages-Bengali, Gujarati, Hindi, Kannada, Malayalam, Tamil, and Telugu. We introduce a multilingual riddle dataset combining traditional riddles with context-reconstructed variants and evaluate five LLMs-Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, and LLaMA 4 Maverick-under seven prompting strategies. In the first stage, we assess riddle-solving performance and find that while Gemini 2.5 Pro performs best overall, few-shot methods yield only marginal gains, and accuracy varies notably across languages. In the second stage, we conduct a self-evaluation experiment to measure reasoning consistency. The results reveal a key finding: a model's initial accuracy is inversely correlated with its ability to identify its own mistakes. Top-performing models such as Gemini 2.5 Pro are overconfident (4.34% True Negative Rate), whereas lower-performing models like LLaMA 4 Scout are substantially more self-aware (42.09% True Negative Rate). These results point to clear gaps in multilingual reasoning and highlight the need for models that not only reason effectively but also recognize their own limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24450v2">Charting the European LLM Benchmarking Landscape: A New Taxonomy and a Set of Best Practices</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 17 pages, 1 figure, 4 tables. Submitted to the LREC 2026 conference
    </div>
    <details class="paper-abstract">
      While new benchmarks for large language models (LLMs) are being developed continuously to catch up with the growing capabilities of new models and AI in general, using and evaluating LLMs in non-English languages remains a little-charted landscape. We give a concise overview of recent developments in LLM benchmarking, and then propose a new taxonomy for the categorization of benchmarks that is tailored to multilingual or non-English use scenarios. We further propose a set of best practices and quality standards that could lead to a more coordinated development of benchmarks for European languages. Among other recommendations, we advocate for a higher language and culture sensitivity of evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02521v1">Large Lemma Miners: Can LLMs do Induction Proofs for Hardware?</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown potential for solving mathematical tasks. We show that LLMs can be utilized to generate proofs by induction for hardware verification and thereby replace some of the manual work done by Formal Verification engineers and deliver industrial value. We present a neurosymbolic approach that includes two prompting frameworks to generate candidate invariants, which are checked using a formal, symbolic tool. Our results indicate that with sufficient reprompting, LLMs are able to generate inductive arguments for mid-size open-source RTL designs. For $87\%$ of our problem set, at least one of the prompt setups succeeded in producing a provably correct inductive argument.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00926v2">LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 19 pages, 6 figures, 28 models tested across 4,200 trials
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) grow in capability, do they develop self-awareness as an emergent behavior? And if so, can we measure it? We introduce the AI Self-Awareness Index (AISAI), a game-theoretic framework for measuring self-awareness through strategic differentiation. Using the "Guess 2/3 of Average" game, we test 28 models (OpenAI, Anthropic, Google) across 4,200 trials with three opponent framings: (A) against humans, (B) against other AI models, and (C) against AI models like you. We operationalize self-awareness as the capacity to differentiate strategic reasoning based on opponent type. Finding 1: Self-awareness emerges with model advancement. The majority of advanced models (21/28, 75%) demonstrate clear self-awareness, while older/smaller models show no differentiation. Finding 2: Self-aware models rank themselves as most rational. Among the 21 models with self-awareness, a consistent rationality hierarchy emerges: Self > Other AIs > Humans, with large AI attribution effects and moderate self-preferencing. These findings reveal that self-awareness is an emergent capability of advanced LLMs, and that self-aware models systematically perceive themselves as more rational than humans. This has implications for AI alignment, human-AI collaboration, and understanding AI beliefs about human capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00847v2">Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $\epsilon\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-\epsilon}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23081v2">A Survey on LLM Mid-Training</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have highlighted the significant benefits of multi-stage training, with a particular emphasis on the emergence of mid-training as a vital stage that bridges pre-training and post-training. Mid-training is distinguished by its use of intermediate data and computational resources, systematically enhancing specified capabilities such as mathematics, coding, reasoning, and long-context extension, while maintaining foundational competencies. This survey provides a formal definition of mid-training for large language models (LLMs) and investigates optimization frameworks that encompass data curation, training strategies, and model architecture optimization. We analyze mainstream model implementations in the context of objective-driven interventions, illustrating how mid-training serves as a distinct and critical stage in the progressive development of LLM capabilities. By clarifying the unique contributions of mid-training, this survey offers a comprehensive taxonomy and actionable insights, supporting future research and innovation in the advancement of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02469v1">Modeling Hawkish-Dovish Latent Beliefs in Multi-Agent Debate-Based LLMs for Monetary Policy Decision Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 PRIMA2025 Accepted
    </div>
    <details class="paper-abstract">
      Accurately forecasting central bank policy decisions, particularly those of the Federal Open Market Committee(FOMC) has become increasingly important amid heightened economic uncertainty. While prior studies have used monetary policy texts to predict rate changes, most rely on static classification models that overlook the deliberative nature of policymaking. This study proposes a novel framework that structurally imitates the FOMC's collective decision-making process by modeling multiple large language models(LLMs) as interacting agents. Each agent begins with a distinct initial belief and produces a prediction based on both qualitative policy texts and quantitative macroeconomic indicators. Through iterative rounds, agents revise their predictions by observing the outputs of others, simulating deliberation and consensus formation. To enhance interpretability, we introduce a latent variable representing each agent's underlying belief(e.g., hawkish or dovish), and we theoretically demonstrate how this belief mediates the perception of input information and interaction dynamics. Empirical results show that this debate-based approach significantly outperforms standard LLMs-based baselines in prediction accuracy. Furthermore, the explicit modeling of beliefs provides insights into how individual perspectives and social influence shape collective policy forecasts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02458v1">Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 9 pages, 8-pages appendix, accepted at ICAIF 25
    </div>
    <details class="paper-abstract">
      We evaluate whether persona-based prompting improves Large Language Model (LLM) performance on macroeconomic forecasting tasks. Using 2,368 economics-related personas from the PersonaHub corpus, we prompt GPT-4o to replicate the ECB Survey of Professional Forecasters across 50 quarterly rounds (2013-2025). We compare the persona-prompted forecasts against the human experts panel, across four target variables (HICP, core HICP, GDP growth, unemployment) and four forecast horizons. We also compare the results against 100 baseline forecasts without persona descriptions to isolate its effect. We report two main findings. Firstly, GPT-4o and human forecasters achieve remarkably similar accuracy levels, with differences that are statistically significant yet practically modest. Our out-of-sample evaluation on 2024-2025 data demonstrates that GPT-4o can maintain competitive forecasting performance on unseen events, though with notable differences compared to the in-sample period. Secondly, our ablation experiment reveals no measurable forecasting advantage from persona descriptions, suggesting these prompt components can be omitted to reduce computational costs without sacrificing accuracy. Our results provide evidence that GPT-4o can achieve competitive forecasting accuracy even on out-of-sample macroeconomic events, if provided with relevant context data, while revealing that diverse prompts produce remarkably homogeneous forecasts compared to human panels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02451v1">Merging Continual Pretraining Models for Domain-Specialized LLMs: A Case Study in Finance</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      While LLMs excel at general tasks, they struggle in specialized domains like finance, requiring diverse skills in domain knowledge, mathematical reasoning, and multilingual processing. Merging domain-specific Continual Pre-training (CPT) "experts" offers a practical alternative to costly and unstable multi-skill training. However, unlike established Supervised Fine-Tuning (SFT) model-based merging, CPT model merging remains largely unexplored. We address this gap by creating financial LLMs from experts in finance, math, and Japanese. We propose a three-stage evaluation focusing on knowledge recovery, complementarity, and emergence, and assess three merging methods (Task Arithmetic, TIES, and DARE-TIES) on a comprehensive financial benchmark curated from 18 tasks across 8 established datasets. Results show that merging an expert with its base model recovers general knowledge lost during CPT, while merging experts improves performance and can yield emergent cross-domain skills. Among the methods, Task Arithmetic performs strongly but is hyperparameter-sensitive, whereas TIES is more robust. Our findings also suggest that while model similarity correlates with merging success, emergent skills depend on more complex factors. This work presents the first foundational analysis of CPT model merging, establishing a principled framework and providing clear guidance for building multi-skill LLMs from existing assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v5">The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce security vulnerabilities that extend beyond traditional content generation to system-level compromises. This paper presents a comprehensive evaluation of the LLMs security used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving computer takeovers. We focus on how different attack surfaces and trust boundaries can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection, and 83.3% are vulnerable to the more stealthy and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed that LLMs which successfully resist direct injection or RAG backdoor attacks will execute identical payloads when requested by peer agents. We found that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks, and that every model exhibits context-dependent security behaviors that create exploitable blind spots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02434v1">Who's Who? LLM-assisted Software Traceability with Architecture Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Identifying architecturally relevant entities in textual artifacts is crucial for Traceability Link Recovery (TLR) between Software Architecture Documentation (SAD) and source code. While Software Architecture Models (SAMs) can bridge the semantic gap between these artifacts, their manual creation is time-consuming. Large Language Models (LLMs) offer new capabilities for extracting architectural entities from SAD and source code to construct SAMs automatically or establish direct trace links. This paper presents two LLM-based approaches: ExArch extracts component names as simple SAMs from SAD and source code to eliminate the need for manual SAM creation, while ArTEMiS identifies architectural entities in documentation and matches them with (manually or automatically generated) SAM entities. Our evaluation compares against state-of-the-art approaches SWATTR, TransArC and ArDoCode. TransArC achieves strong performance (F1: 0.87) but requires manually created SAMs; ExArch achieves comparable results (F1: 0.86) using only SAD and code. ArTEMiS is on par with the traditional heuristic-based SWATTR (F1: 0.81) and can successfully replace it when integrated with TransArC. The combination of ArTEMiS and ExArch outperforms ArDoCode, the best baseline without manual SAMs. Our results demonstrate that LLMs can effectively identify architectural entities in textual artifacts, enabling automated SAM generation and TLR, making architecture-code traceability more practical and accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02424v1">ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled significant progress in decision-making and task planning for embodied autonomous agents. However, most existing methods still struggle with complex, long-horizon tasks because they rely on a monolithic trajectory that entangles all past decisions and observations, attempting to solve the entire task in a single unified process. To address this limitation, we propose ReAcTree, a hierarchical task-planning method that decomposes a complex goal into more manageable subgoals within a dynamically constructed agent tree. Each subgoal is handled by an LLM agent node capable of reasoning, acting, and further expanding the tree, while control flow nodes coordinate the execution strategies of agent nodes. In addition, we integrate two complementary memory systems: each agent node retrieves goal-specific, subgoal-level examples from episodic memory and shares environment-specific observations through working memory. Experiments on the WAH-NL and ALFRED datasets demonstrate that ReAcTree consistently outperforms strong task-planning baselines such as ReAct across diverse LLMs. Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5 72B, nearly doubling ReAct's 31%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02399v1">EvoDev: An Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advances in large language model agents offer the promise of automating end-to-end software development from natural language requirements. However, existing approaches largely adopt linear, waterfall-style pipelines, which oversimplify the iterative nature of real-world development and struggle with complex, large-scale projects. To address these limitations, we propose EvoDev, an iterative software development framework inspired by feature-driven development. EvoDev decomposes user requirements into a set of user-valued features and constructs a Feature Map, a directed acyclic graph that explicitly models dependencies between features. Each node in the feature map maintains multi-level information, including business logic, design, and code, which is propagated along dependencies to provide context for subsequent development iterations. We evaluate EvoDev on challenging Android development tasks and show that it outperforms the best-performing baseline, Claude Code, by a substantial margin of 56.8%, while improving single-agent performance by 16.0%-76.6% across different base LLMs, highlighting the importance of dependency modeling, context propagation, and workflow-aware agent design for complex software projects. Our work summarizes practical insights for designing iterative, LLM-driven development frameworks and informs future training of base LLMs to better support iterative software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02378v1">Revisiting put-that-there, context aware window interactions via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      We revisit Bolt's classic "Put-That-There" concept for modern head-mounted displays by pairing Large Language Models (LLMs) with XR sensor and tech stack. The agent fuses (i) a semantically segmented 3-D environment, (ii) live application metadata, and (iii) users' verbal, pointing, and head-gaze cues to issue JSON window-placement actions. As a result, users can manage a panoramic workspace through: (1) explicit commands ("Place Google Maps on the coffee table"), (2) deictic speech plus gestures ("Put that there"), or (3) high-level goals ("I need to send a message"). Unlike traditional explicit interfaces, our system supports one-to-many action mappings and goal-centric reasoning, allowing the LLM to dynamically infer relevant applications and layout decisions, including interrelationships across tools. This enables seamless, intent-driven interaction without manual window juggling in immersive XR environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15207v3">FlowRL: Matching Reward Distributions for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      We propose FlowRL: matching the full reward distribution via flow balancing instead of maximizing rewards in large language model (LLM) reinforcement learning (RL). Recent advanced reasoning models adopt reward-maximizing methods (\eg, PPO and GRPO), which tend to over-optimize dominant reward signals while neglecting less frequent but valid reasoning paths, thus reducing diversity. In contrast, we transform scalar rewards into a normalized target distribution using a learnable partition function, and then minimize the reverse KL divergence between the policy and the target distribution. We implement this idea as a flow-balanced optimization method that promotes diverse exploration and generalizable reasoning trajectories. We conduct experiments on math and code reasoning tasks: FlowRL achieves a significant average improvement of $10.0\%$ over GRPO and $5.1\%$ over PPO on math benchmarks, and performs consistently better on code reasoning tasks. These results highlight reward distribution-matching as a key step toward efficient exploration and diverse reasoning in LLM reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02366v1">LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      In this work, we propose LiveSecBench, a dynamic and continuously updated safety benchmark specifically for Chinese-language LLM application scenarios. LiveSecBench evaluates models across six critical dimensions (Legality, Ethics, Factuality, Privacy, Adversarial Robustness, and Reasoning Safety) rooted in the Chinese legal and social frameworks. This benchmark maintains relevance through a dynamic update schedule that incorporates new threat vectors, such as the planned inclusion of Text-to-Image Generation Safety and Agentic Safety in the next update. For now, LiveSecBench (v251030) has evaluated 18 LLMs, providing a landscape of AI safety in the context of Chinese language. The leaderboard is publicly accessible at https://livesecbench.intokentech.cn/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02356v1">An Automated Framework for Strategy Discovery, Retrieval, and Evolution in LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      The widespread deployment of Large Language Models (LLMs) as public-facing web services and APIs has made their security a core concern for the web ecosystem. Jailbreak attacks, as one of the significant threats to LLMs, have recently attracted extensive research. In this paper, we reveal a jailbreak strategy which can effectively evade current defense strategies. It can extract valuable information from failed or partially successful attack attempts and contains self-evolution from attack interactions, resulting in sufficient strategy diversity and adaptability. Inspired by continuous learning and modular design principles, we propose ASTRA, a jailbreak framework that autonomously discovers, retrieves, and evolves attack strategies to achieve more efficient and adaptive attacks. To enable this autonomous evolution, we design a closed-loop "attack-evaluate-distill-reuse" core mechanism that not only generates attack prompts but also automatically distills and generalizes reusable attack strategies from every interaction. To systematically accumulate and apply this attack knowledge, we introduce a three-tier strategy library that categorizes strategies into Effective, Promising, and Ineffective based on their performance scores. The strategy library not only provides precise guidance for attack generation but also possesses exceptional extensibility and transferability. We conduct extensive experiments under a black-box setting, and the results show that ASTRA achieves an average Attack Success Rate (ASR) of 82.7%, significantly outperforming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02303v1">Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09586v3">ValueCompass: A Framework for Measuring Contextual Value Alignment Between Human and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      As AI systems become more advanced, ensuring their alignment with a diverse range of individuals and societal values becomes increasingly critical. But how can we capture fundamental human values and assess the degree to which AI systems align with them? We introduce ValueCompass, a framework of fundamental values, grounded in psychological theory and a systematic review, to identify and evaluate human-AI alignment. We apply ValueCompass to measure the value alignment of humans and large language models (LLMs) across four real-world scenarios: collaborative writing, education, public sectors, and healthcare. Our findings reveal concerning misalignments between humans and LLMs, such as humans frequently endorse values like "National Security" which were largely rejected by LLMs. We also observe that values differ across scenarios, highlighting the need for context-aware AI alignment strategies. This work provides valuable insights into the design space of human-AI alignment, laying the foundations for developing AI systems that responsibly reflect societal values and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00032v2">Strategic Communication and Language Bias in Multi-Agent LLM Coordination</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents are increasingly deployed in multi-agent scenarios where coordination is crucial but not always assured. Research shows that the way strategic scenarios are framed linguistically can affect cooperation. This paper explores whether allowing agents to communicate amplifies these language-driven effects. Leveraging FAIRGAME, we simulate one-shot and repeated games across different languages and models, both with and without communication. Our experiments, conducted with two advanced LLMs-GPT-4o and Llama 4 Maverick-reveal that communication significantly influences agent behavior, though its impact varies by language, personality, and game structure. These findings underscore the dual role of communication in fostering coordination and reinforcing biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02805v1">MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Project page: https://github.com/icip-cas/MemSearcher
    </div>
    <details class="paper-abstract">
      Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at https://github.com/icip-cas/MemSearcher
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02795v1">Can LLMs subtract numbers?</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Work-in-progress; MathNLP non-archival presentation
    </div>
    <details class="paper-abstract">
      We present a systematic study of subtraction in large language models (LLMs). While prior benchmarks emphasize addition and multiplication, subtraction has received comparatively little attention despite being structurally distinct as a non-commutative operation. We evaluate eight pretrained LLMs spanning four families on addition and subtraction problems. Our experiments reveal that subtraction accuracy lags behind addition by a wide margin. We find that the errors for ($a-b$) are concentrated in cases where ($a<b$). In such cases, LLMs frequently produce the correct magnitude but omit the negative sign. Probing analyses show that LLMs internally encode whether results should be negative, yet this information is often not reflected in generated outputs. We further test well-known techniques such as few-shot learning and instruction-tuning to see if they can improve the LLMs' performance. Our results suggest that while few-shot prompting yields modest gains, the instruction-tuned models achieve near-perfect accuracies in generating the negative sign. Together, these findings provide a clearer characterization of the limitations and recoverability of LLMs' arithmetic capabilities in subtraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00130v2">A Comparative Analysis of LLM Adaptation: SFT, LoRA, and ICL in Data-Scarce Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      The remarkable capabilities of Large Language Models (LLMs) often need to be tailored for specific applications, requiring the integration of new knowledge or the acquisition of new skills. While full fine-tuning is a powerful adaptation method, it is computationally expensive and can lead to a degradation of general reasoning abilities, a phenomenon known as catastrophic forgetting. A range of alternative techniques exists, each with its own trade-offs. In-Context Learning (ICL) is fast but limited by context length, while Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA) offer a middle ground by minimizing parameter changes. However, the challenge of catastrophic forgetting persists, raising questions about the best adaptation strategy for a given task. This paper presents a comparative analysis of Supervised Finetuning (SFT), LoRA, and ICL in data-scarce scenarios. We find that LoRA provides the most effective balance, successfully instilling new skills with minimal impact on the base model's general knowledge. In contrast, while SFT excels at skill acquisition, it is highly susceptible to catastrophic forgetting. ICL is effective for incorporating factual knowledge but struggles with complex skills. Our findings offer a practical framework for selecting an LLM adaptation strategy. We highlight the critical distinction between skill acquisition and knowledge integration, clarify the trade-offs between task-specific performance and the preservation of general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02759v1">LLM-Supported Formal Knowledge Representation for Enhancing Control Engineering Content with an Interactive Semantic Layer</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of research output in control engineering calls for new approaches to structure and formalize domain knowledge. This paper briefly describes an LLM-supported method for semi-automated generation of formal knowledge representations that combine human readability with machine interpretability and increased expressiveness. Based on the Imperative Representation of Knowledge (PyIRK) framework, we demonstrate how language models can assist in transforming natural-language descriptions and mathematical definitions (available as LaTeX source code) into a formalized knowledge graph. As a first application we present the generation of an ``interactive semantic layer'' to enhance the source documents in order to facilitate knowledge transfer. From our perspective this contributes to the vision of easily accessible, collaborative, and verifiable knowledge bases for the control engineering domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02755v1">Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit complementary strengths across domains and come with varying inference costs, motivating the design of multi-agent LLM systems where specialized models collaborate efficiently. Existing approaches predominantly rely on decentralized frameworks, which invoke multiple LLMs for every input and thus lead to substantial and uncontrolled inference costs. In this work, we introduce a centralized multi-LLM framework, where a controller LLM selectively coordinates a pool of expert models in a cost-efficient and cost-controllable manner. We formulate this coordination problem as reinforcement learning with dual objectives: maximizing task performance while minimizing the overall inference cost. In addition, we expect the multi-agent system to have adapted behavior with different budget conditions during inference. To this end, we propose CoRL, a reinforcement learning framework that optimizes the performance cost trade-off in a controllable multi-budget setting. Experiments on four diverse benchmarks demonstrate that CoRL enables a single system to surpass the best expert LLM under high-budget settings, while maintaining strong performance in more economical low-budget modes, highlighting the effectiveness of centralized coordination for scalable and cost-efficient multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02734v1">CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents' ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents' economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and necessitate agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than 75% exact match rate on the hardest tasks, and performance further dropping by around 40% under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16116v3">DMind Benchmark: Toward a Holistic Assessment of LLM Capabilities across the Web3 Domain</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive performance in diverse natural language processing tasks, but specialized domains such as Web3 present new challenges and require more tailored evaluation. Despite the significant user base and capital flows in Web3, encompassing smart contracts, decentralized finance (DeFi), non-fungible tokens (NFTs), decentralized autonomous organizations (DAOs), on-chain governance, and novel token-economics, no comprehensive benchmark has systematically assessed LLM performance in this domain. To address this gap, we introduce the DMind Benchmark, a holistic Web3-oriented evaluation suite covering nine critical subfields: fundamental blockchain concepts, blockchain infrastructure, smart contract, DeFi mechanisms, DAOs, NFTs, token economics, meme concept, and security vulnerabilities. Beyond multiple-choice questions, DMind Benchmark features domain-specific tasks such as contract debugging and on-chain numeric reasoning, mirroring real-world scenarios. We evaluated 26 models, including ChatGPT, Claude, DeepSeek, Gemini, Grok, and Qwen, uncovering notable performance gaps in specialized areas like token economics and security-critical contract analysis. While some models excel in blockchain infrastructure tasks, advanced subfields remain challenging. Our benchmark dataset and evaluation pipeline are open-sourced on https://huggingface.co/datasets/DMindAI/DMind_Benchmark, reaching number one in Hugging Face's trending dataset charts within a week of release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01854v2">Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent advances in LLM Multi-Agent Systems enable scalable orchestration of sub-agents, each coordinating hundreds or thousands of tools or Model Context Protocol (MCP) servers. However, existing retrieval methods typically match queries against coarse agent-level descriptions before routing, which obscures fine-grained tool functionality and often results in suboptimal agent selection. We introduce Tool-to-Agent Retrieval, a unified framework that embeds both tools and their parent agents in a shared vector space and connects them through metadata relationships. By explicitly representing tool capabilities and traversing metadata to the agent level, Tool-to-Agent Retrieval enables granular tool-level or agent-level retrieval, ensuring that agents and their underlying tools or MCP servers are equally represented without the context dilution that arises from chunking many tools together. Evaluating Tool-to-Agent Retrieval across eight embedding models, our approach achieves consistent improvements of 19.4% in Recall@5 and 17.7% in nDCG@5 over previous state-of-the-art agent retrievers on the LiveMCPBench benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04705v1">POLIS-Bench: Towards Multi-Dimensional Evaluation of LLMs for Bilingual Policy Tasks in Governmental Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      We introduce POLIS-Bench, the first rigorous, systematic evaluation suite designed for LLMs operating in governmental bilingual policy scenarios. Compared to existing benchmarks, POLIS-Bench introduces three major advancements. (i) Up-to-date Bilingual Corpus: We construct an extensive, up-to-date policy corpus that significantly scales the effective assessment sample size, ensuring relevance to current governance practice. (ii) Scenario-Grounded Task Design: We distill three specialized, scenario-grounded tasks -- Clause Retrieval & Interpretation, Solution Generation, and the Compliance Judgmen--to comprehensively probe model understanding and application. (iii) Dual-Metric Evaluation Framework: We establish a novel dual-metric evaluation framework combining semantic similarity with accuracy rate to precisely measure both content alignment and task requirement adherence. A large-scale evaluation of over 10 state-of-the-art LLMs on POLIS-Bench reveals a clear performance hierarchy where reasoning models maintain superior cross-task stability and accuracy, highlighting the difficulty of compliance tasks. Furthermore, leveraging our benchmark, we successfully fine-tune a lightweight open-source model. The resulting POLIS series models achieves parity with, or surpasses, strong proprietary baselines on multiple policy subtasks at a significantly reduced cost, providing a cost-effective and compliant path for robust real-world governmental deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.00203v1">Diffusion LLMs are Natural Adversaries for any LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      We introduce a novel framework that transforms the resource-intensive (adversarial) prompt optimization problem into an \emph{efficient, amortized inference task}. Our core insight is that pretrained, non-autoregressive generative LLMs, such as Diffusion LLMs, which model the joint distribution over prompt-response pairs, can serve as powerful surrogates for prompt search. This approach enables the direct conditional generation of prompts, effectively replacing costly, per-instance discrete optimization with a small number of parallelizable samples. We provide a probabilistic analysis demonstrating that under mild fidelity assumptions, only a few conditional samples are required to recover high-reward (harmful) prompts. Empirically, we find that the generated prompts are low-perplexity, diverse jailbreaks that exhibit strong transferability to a wide range of black-box target models, including robustly trained and proprietary LLMs. Beyond adversarial prompting, our framework opens new directions for red teaming, automated prompt optimization, and leveraging emerging Flow- and Diffusion-based LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02795v1">Can LLMs subtract numbers?</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Work-in-progress; MathNLP non-archival presentation
    </div>
    <details class="paper-abstract">
      We present a systematic study of subtraction in large language models (LLMs). While prior benchmarks emphasize addition and multiplication, subtraction has received comparatively little attention despite being structurally distinct as a non-commutative operation. We evaluate eight pretrained LLMs spanning four families on addition and subtraction problems. Our experiments reveal that subtraction accuracy lags behind addition by a wide margin. We find that the errors for ($a-b$) are concentrated in cases where ($a<b$). In such cases, LLMs frequently produce the correct magnitude but omit the negative sign. Probing analyses show that LLMs internally encode whether results should be negative, yet this information is often not reflected in generated outputs. We further test well-known techniques such as few-shot learning and instruction-tuning to see if they can improve the LLMs' performance. Our results suggest that while few-shot prompting yields modest gains, the instruction-tuned models achieve near-perfect accuracies in generating the negative sign. Together, these findings provide a clearer characterization of the limitations and recoverability of LLMs' arithmetic capabilities in subtraction.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02135v1">Rethinking LLM Human Simulation: When a Graph is What You Need</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 Code: https://github.com/schang-lab/gems
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate humans, with applications ranging from survey prediction to decision-making. However, are LLMs strictly necessary, or can smaller, domain-grounded models suffice? We identify a large class of simulation problems in which individuals make choices among discrete options, where a graph neural network (GNN) can match or surpass strong LLM baselines despite being three orders of magnitude smaller. We introduce Graph-basEd Models for human Simulation (GEMS), which casts discrete choice simulation tasks as a link prediction problem on graphs, leveraging relational knowledge while incorporating language representations only when needed. Evaluations across three key settings on three simulation datasets show that GEMS achieves comparable or better accuracy than LLMs, with far greater efficiency, interpretability, and transparency, highlighting the promise of graph-based modeling as a lightweight alternative to LLMs for human simulation. Our code is available at https://github.com/schang-lab/gems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17013v2">DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16891v2">LLMs as Layout Designers: Enhanced Spatial Reasoning for Content-Aware Layout Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their ability to understand and manipulate spatial relationships remains limited. Such capabilities are crucial for content-aware graphic layout design, where the goal is to arrange heterogeneous elements onto a canvas so that final design remains visually balanced and structurally feasible. This problem requires precise coordination of placement, alignment, and structural organization of multiple elements within a constrained visual space. To address this limitation, we introduce LaySPA, a reinforcement learning-based framework that augments LLM-based agents with explicit spatial reasoning capabilities for layout design. LaySPA employs hybrid reward signals that jointly capture geometric constraints, structural fidelity, and visual quality, enabling agents to navigate the canvas, model inter-element relationships, and optimize spatial arrangements. Through group-relative policy optimization, the agent generates content-aware layouts that reflect salient regions, respect spatial constraints, and produces an interpretable reasoning trace explaining placement decisions and a structured layout specification. Experimental results show that LaySPA substantially improves the generation of structurally valid and visually appealing layouts, outperforming larger general-purpose LLMs and achieving performance comparable to state-of-the-art specialized layout models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v5">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 An earlier version of this paper was published in AutoML 2025 Methods Track. This version adds missing standard deviations in Table 1
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01850v1">SmartMLOps Studio: Design of an LLM-Integrated IDE with Automated MLOps Pipelines for Model Development and Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      The rapid expansion of artificial intelligence and machine learning (ML) applications has intensified the demand for integrated environments that unify model development, deployment, and monitoring. Traditional Integrated Development Environments (IDEs) focus primarily on code authoring, lacking intelligent support for the full ML lifecycle, while existing MLOps platforms remain detached from the coding workflow. To address this gap, this study proposes the design of an LLM-Integrated IDE with automated MLOps pipelines that enables continuous model development and monitoring within a single environment. The proposed system embeds a Large Language Model (LLM) assistant capable of code generation, debugging recommendation, and automatic pipeline configuration. The backend incorporates automated data validation, feature storage, drift detection, retraining triggers, and CI/CD deployment orchestration. This framework was implemented in a prototype named SmartMLOps Studio and evaluated using classification and forecasting tasks on the UCI Adult and M5 datasets. Experimental results demonstrate that SmartMLOps Studio reduces pipeline configuration time by 61%, improves experiment reproducibility by 45%, and increases drift detection accuracy by 14% compared to traditional workflows. By bridging intelligent code assistance and automated operational pipelines, this research establishes a novel paradigm for AI engineering - transforming the IDE from a static coding tool into a dynamic, lifecycle-aware intelligent platform for scalable and efficient model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01840v1">A Detailed Study on LLM Biases Concerning Corporate Social Responsibility and Green Supply Chains</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 37 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Organizations increasingly use Large Language Models (LLMs) to improve supply chain processes and reduce environmental impacts. However, LLMs have been shown to reproduce biases regarding the prioritization of sustainable business strategies. Thus, it is important to identify underlying training data biases that LLMs pertain regarding the importance and role of sustainable business and supply chain practices. This study investigates how different LLMs respond to validated surveys about the role of ethics and responsibility for businesses, and the importance of sustainable practices and relations with suppliers and customers. Using standardized questionnaires, we systematically analyze responses generated by state-of-the-art LLMs to identify variations. We further evaluate whether differences are augmented by four organizational culture types, thereby evaluating the practical relevance of identified biases. The findings reveal significant systematic differences between models and demonstrate that organizational culture prompts substantially modify LLM responses. The study holds important implications for LLM-assisted decision-making in sustainability contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01815v1">KV Cache Transform Coding for Compact Storage in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy, and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks including AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, and MATH-500. It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01807v1">Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 Presented at Workshop on Prompt Optimization, KDD 2025, Toronto, Canada
    </div>
    <details class="paper-abstract">
      Length control in Large Language Models (LLMs) is a crucial but under-addressed challenge, with applications ranging from voice interfaces requiring concise responses to research summaries needing comprehensive outputs. Current approaches to length control, including Regularized DPO, Length-Instruction Fine Tuning, and tool-augmented methods, typically require expensive model retraining or complex inference-time tooling. This paper presents a prompt engineering methodology that enables precise length control without model retraining. Our structure-guided approach implements deliberate planning and word counting mechanisms within the prompt, encouraging the model to carefully track and adhere to specified length constraints. Comprehensive evaluations across six state-of-the-art LLMs demonstrate that our method significantly improves length fidelity for several models compared to standard prompting when applied to document summarization tasks, particularly for shorter-to-medium length constraints. The proposed technique shows varying benefits across different model architectures, with some models demonstrating up to 37.6% improvement in length adherence. Quality evaluations further reveal that our approach maintains or enhances overall output quality compared to standard prompting techniques. Our approach provides an immediately deployable solution for applications requiring precise length control, particularly valuable for production environments where model retraining is impractical or cost-prohibitive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01757v1">Towards LLM-Powered Task-Aware Retrieval of Scientific Workflows for Galaxy</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Scientific Workflow Management Systems (SWfMSs) such as Galaxy have become essential infrastructure in bioinformatics, supporting the design, execution, and sharing of complex multi-step analyses. Despite hosting hundreds of reusable workflows across domains, Galaxy's current keyword-based retrieval system offers limited support for semantic query interpretation and often fails to surface relevant workflows when exact term matches are absent. To address this gap, we propose a task-aware, two-stage retrieval framework that integrates dense vector search with large language model (LLM)-based reranking. Our system first retrieves candidate workflows using state-of-the-art embedding models and then reranks them using instruction-tuned generative LLMs (GPT-4o, Mistral-7B) based on semantic task alignment. To support robust evaluation, we construct a benchmark dataset of Galaxy workflows annotated with semantic topics via BERTopic and synthesize realistic task-oriented queries using LLMs. We conduct a comprehensive comparison of lexical, dense, and reranking models using standard IR metrics, presenting the first systematic evaluation of retrieval performance in the Galaxy ecosystem. Results show that our approach significantly improves top-k accuracy and relevance, particularly for long or under-specified queries. We further integrate our system as a prototype tool within Galaxy, providing a proof-of-concept for LLM-enhanced workflow search. This work advances the usability and accessibility of scientific workflows, especially for novice users and interdisciplinary researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01746v1">Scam Shield: Multi-Model Voting and Fine-Tuned LLMs Against Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Scam detection remains a critical challenge in cybersecurity as adversaries craft messages that evade automated filters. We propose a Hierarchical Scam Detection System (HSDS) that combines a lightweight multi-model voting front end with a fine-tuned LLaMA 3.1 8B Instruct back end to improve accuracy and robustness against adversarial attacks. An ensemble of four classifiers provides preliminary predictions through majority vote, and ambiguous cases are escalated to the fine-tuned model, which is optimized with adversarial training to reduce misclassification. Experiments show that this hierarchical design both improves adversarial scam detection and shortens inference time by routing most cases away from the LLM, outperforming traditional machine-learning baselines and proprietary LLM baselines. The findings highlight the effectiveness of a hybrid voting mechanism and adversarial fine-tuning in fortifying LLMs against evolving scam tactics, enhancing the resilience of automated scam detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06289v4">Automate Strategy Finding with LLM in Quant Investment</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      We present a novel three-stage framework leveraging Large Language Models (LLMs) within a risk-aware multi-agent system for automate strategy finding in quantitative finance. Our approach addresses the brittleness of traditional deep learning models in financial applications by: employing prompt-engineered LLMs to generate executable alpha factor candidates across diverse financial data, implementing multimodal agent-based evaluation that filters factors based on market status, predictive quality while maintaining category balance, and deploying dynamic weight optimization that adapts to market conditions. Experimental results demonstrate the robust performance of the strategy in Chinese & US market regimes compared to established benchmarks. Our work extends LLMs capabilities to quantitative trading, providing a scalable architecture for financial signal extraction and portfolio construction. The overall framework significantly outperforms all benchmarks with 53.17% cumulative return on SSE50 (Jan 2023 to Jan 2024), demonstrating superior risk-adjusted performance and downside protection on the market.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01633v1">Scaling Graph Chain-of-Thought Reasoning: A Multi-Agent Framework with Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Graph Chain-of-Thought (Graph-CoT) enables large language models (LLMs) to perform step-by-step reasoning over graph-structured knowledge, but existing pipelines suffer from low accuracy, excessive token usage, high latency, and low throughput due to single-agent monolithic prompts, repeated context re-encoding, and inefficient serving execution. We present GLM, the first multi-agent Graph-CoT system co-designed with an optimized LLM serving architecture. GLM decomposes reasoning into specialized agents for classification, reasoning, action generation, and graph retrieval, enabling branching and selective context sharing to reduce prompt length and reasoning iterations while preserving reasoning quality, thereby improving accuracy and reducing overall token consumption. To scale inference, we introduce a Graph-CoT-aware LLM inference mechanism with graph-specific KV-cache management, priority-based eviction, and pipelined execution to improve serving efficiency. Experiments demonstrate that GLM improves answer accuracy by up to 38%, reduces token cost by up to 95.7%, lowers inference latency by 90.3%, and achieves up to 15.1x higher throughput compared to state-of-the-art Graph-CoT baselines, enabling efficient adoption for complex real-world reasoning at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01602v1">L2T-Tune:LLM-Guided Hybrid Database Tuning with LHS and TD3</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Configuration tuning is critical for database performance. Although recent advancements in database tuning have shown promising results in throughput and latency improvement, challenges remain. First, the vast knob space makes direct optimization unstable and slow to converge. Second, reinforcement learning pipelines often lack effective warm-start guidance and require long offline training. Third, transferability is limited: when hardware or workloads change, existing models typically require substantial retraining to recover performance. To address these limitations, we propose L2T-Tune, a new LLM-guided hybrid database tuning framework that features a three-stage pipeline: Stage one performs a warm start that simultaneously generates uniform samples across the knob space and logs them into a shared pool; Stage two leverages a large language model to mine and prioritize tuning hints from manuals and community documents for rapid convergence. Stage three uses the warm-start sample pool to reduce the dimensionality of knobs and state features, then fine-tunes the configuration with the Twin Delayed Deep Deterministic Policy Gradient algorithm. We conduct experiments on L2T-Tune and the state-of-the-art models. Compared with the best-performing alternative, our approach improves performance by an average of 37.1% across all workloads, and by up to 73% on TPC-C. Compared with models trained with reinforcement learning, it achieves rapid convergence in the offline tuning stage on a single server. Moreover, during the online tuning stage, it only takes 30 steps to achieve best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01482v1">Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. Our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01454v1">"Don't Teach Minerva": Guiding LLMs Through Complex Syntax for Faithful Latin Translation with RAG</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Translating a morphology-rich, low-resource language like Latin poses significant challenges. This paper introduces a reproducible draft-based refinement pipeline that elevates open-source Large Language Models (LLMs) to a performance level statistically comparable to top-tier proprietary systems. Our method first uses a fine-tuned NLLB-1.3B model to generate a high-quality, structurally faithful draft. A zero-shot LLM (Llama-3.3 or Qwen3) then polishes this draft, a process that can be further enhanced by augmenting the context with retrieved out-context examples (RAG). We demonstrate the robustness of this approach on two distinct benchmarks: a standard in-domain test set (Rosenthal, 2023) and a new, challenging out-of-domain (OOD) set of 12th-century Latin letters (2025). Our central finding is that this open-source RAG system achieves performance statistically comparable to the GPT-5 baseline, without any task-specific LLM fine-tuning. We release the pipeline, the Chartres OOD set, and evaluation scripts and models to facilitate replicability and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01423v1">LLM-Assisted Tool for Joint Generation of Formulas and Functions in Rule-Based Verification of Map Transformations</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      High-definition map transformations are essential in autonomous driving systems, enabling interoperability across tools. Ensuring their semantic correctness is challenging, since existing rule-based frameworks rely on manually written formulas and domain-specific functions, limiting scalability. In this paper, We present an LLM-assisted pipeline that jointly generates logical formulas and corresponding executable predicates within a computational FOL framework, extending the map verifier in CommonRoad scenario designer with elevation support. The pipeline leverages prompt-based LLM generation to produce grammar-compliant rules and predicates that integrate directly into the existing system. We implemented a prototype and evaluated it on synthetic bridge and slope scenarios. The results indicate reduced manual engineering effort while preserving correctness, demonstrating the feasibility of a scalable, semi-automated human-in-the-loop approach to map-transformation verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01375v1">Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 under review, 28 pages
    </div>
    <details class="paper-abstract">
      Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01323v1">DEEPAMBIGQA: Ambiguous Multi-hop Questions for Benchmarking LLM Answer Completeness</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with integrated search tools show strong promise in open-domain question answering (QA), yet they often struggle to produce complete answer set to complex questions such as Which actor from the film Heat won at least one Academy Award?, which requires (1) distinguishing between multiple films sharing the same title and (2) reasoning across a large set of actors to gather and integrate evidence. Existing QA benchmarks rarely evaluate both challenges jointly. To address this, we introduce DeepAmbigQAGen, an automatic data generation pipeline that constructs QA tasks grounded in text corpora and linked knowledge graph, generating natural and verifiable questions that systematically embed name ambiguity and multi-step reasoning. Based on this, we build DeepAmbigQA, a dataset of 3,600 questions requiring multi-hop reasoning and half of them explicit name ambiguity resolving. Experiments reveal that, even state-of-the-art GPT-5 show incomplete answers, achieving only 0.13 exact match on ambiguous questions and 0.21 on non-ambiguous questions. These findings highlight the need for more robust QA systems aimed at information gathering and answer completeness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01311v1">llmSHAP: A Principled Approach to LLM Explainability</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Feature attribution methods help make machine learning-based inference explainable by determining how much one or several features have contributed to a model's output. A particularly popular attribution method is based on the Shapley value from cooperative game theory, a measure that guarantees the satisfaction of several desirable principles, assuming deterministic inference. We apply the Shapley value to feature attribution in large language model (LLM)-based decision support systems, where inference is, by design, stochastic (non-deterministic). We then demonstrate when we can and cannot guarantee Shapley value principle satisfaction across different implementation variants applied to LLM-based decision support, and analyze how the stochastic nature of LLMs affects these guarantees. We also highlight trade-offs between explainable inference speed, agreement with exact Shapley value attributions, and principle attainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01265v1">AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      This paper investigates the impact of domain specificity on abstractive summarisation of Arabic financial texts using large language models (LLMs). We introduce AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning nearly a decade of reporting from October 2015 to July 2025. Designed as the Arabic equivalent of major English summarisation corpora such as CNN/DailyMail, AraFinNews provides a robust benchmark for evaluating domain-specific language understanding and generation in financial contexts. Using this resource, we evaluate transformer-based models -- including mT5, AraT5, and the domain-adapted FinAraT5 -- to examine how financial-domain pretraining influences factual accuracy, numerical reliability, and stylistic alignment with professional reporting. Experimental results show that domain-adapted models generate more faithful and coherent summaries, particularly in handling quantitative and entity-centric information. The findings highlight the importance of domain-specific adaptation for improving factual consistency and narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-UK/AraFinNews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01252v1">Lares: LLM-driven Code Slice Semantic Search for Patch Presence Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      In modern software ecosystems, 1-day vulnerabilities pose significant security risks due to extensive code reuse. Identifying vulnerable functions in target binaries alone is insufficient; it is also crucial to determine whether these functions have been patched. Existing methods, however, suffer from limited usability and accuracy. They often depend on the compilation process to extract features, requiring substantial manual effort and failing for certain software. Moreover, they cannot reliably differentiate between code changes caused by patches or compilation variations. To overcome these limitations, we propose Lares, a scalable and accurate method for patch presence testing. Lares introduces Code Slice Semantic Search, which directly extracts features from the patch source code and identifies semantically equivalent code slices in the pseudocode of the target binary. By eliminating the need for the compilation process, Lares improves usability, while leveraging large language models (LLMs) for code analysis and SMT solvers for logical reasoning to enhance accuracy. Experimental results show that Lares achieves superior precision, recall, and usability. Furthermore, it is the first work to evaluate patch presence testing across optimization levels, architectures, and compilers. The datasets and source code used in this article are available at https://github.com/Siyuan-Li201/Lares.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01248v1">AskNow: An LLM-powered Interactive System for Real-Time Question Answering in Large-Scale Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      In large-scale classrooms, students often struggle to ask questions due to limited instructor attention and social pressure. Based on findings from a formative study with 24 students and 12 instructors, we designed AskNow, an LLM-powered system that enables students to ask questions and receive real-time, context-aware responses grounded in the ongoing lecture and that allows instructors to view students' questions collectively. We deployed AskNow in three university computer science courses and tested with 117 students. To evaluate AskNow's responses, each instructor rated the perceived correctness and satisfaction of 100 randomly sampled AskNow-generated responses. In addition, we conducted interviews with 24 students and the three instructors to understand their experience with AskNow. We found that AskNow significantly reduced students' perceived time to resolve confusion. Instructors rated AskNow's responses as highly accurate and satisfactory. Instructor and student feedback provided insights into supporting real-time learning in large lecture settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01208v1">Contextual Relevance and Adaptive Sampling for LLM-Based Document Reranking</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Reranking algorithms have made progress in improving document retrieval quality by efficiently aggregating relevance judgments generated by large language models (LLMs). However, identifying relevant documents for queries that require in-depth reasoning remains a major challenge. Reasoning-intensive queries often exhibit multifaceted information needs and nuanced interpretations, rendering document relevance inherently context dependent. To address this, we propose contextual relevance, which we define as the probability that a document is relevant to a given query, marginalized over the distribution of different reranking contexts it may appear in (i.e., the set of candidate documents it is ranked alongside and the order in which the documents are presented to a reranking model). While prior works have studied methods to mitigate the positional bias LLMs exhibit by accounting for the ordering of documents, we empirically find that the compositions of these batches also plays an important role in reranking performance. To efficiently estimate contextual relevance, we propose TS-SetRank, a sampling-based, uncertainty-aware reranking algorithm. Empirically, TS-SetRank improves nDCG@10 over retrieval and reranking baselines by 15-25% on BRIGHT and 6-21% on BEIR, highlighting the importance of modeling relevance as context-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01202v1">Forget BIT, It is All about TOKEN: Towards Semantic Information Theory for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in numerous real-world applications. While the vast majority of research conducted from an experimental perspective is progressing rapidly, it demands substantial computational power, data, and other resources. Therefore, how to open the black-box of LLMs from a theoretical standpoint has become a critical challenge. This paper takes the theory of rate-distortion function, directed information, and Granger causality as its starting point to investigate the information-theoretic principles behind LLMs, leading to the development of semantic information theory for LLMs, where the fundamental unit is token, rather than bits that lacks any semantic meaning. By defining the probabilistic model of LLMs, we discuss structure-agnostic information-theoretic measures, such as the directed rate-distortion function in pre-training, the directed rate-reward function in post-training, and the semantic information flow in inference phase. This paper also delves deeply into the theory of token-level semantic embedding and the information-theoretically optimal vectorization method. Thereafter, we propose a general definition of autoregression LLM, where the Transformer architecture and its performance such as ELBO, generalization error bound, memory capacity, and semantic information measures can be derived theoretically. Other architectures, such as Mamba/Mamba2 and LLaDA, are also discussed in our framework. Consequently, this paper provides a theoretical framework for understanding LLMs from the perspective of semantic information theory, which also offers the necessary theoretical tools for further in-depth research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01188v1">ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      The rapid spread of fake news threatens social stability and public trust, rendering its detection an imperative research priority. Although large language models (LLMs) excel at numerous natural language processing tasks with their remarkable contextual understanding and extensive prior knowledge, the time-bounded knowledge coverage and tendency for generating hallucination content reduce their reliability when handling fast-evolving news streams. Furthermore, models trained on existing static datasets also often lack the generalization needed for emerging news topics. To address these challenges, we propose ZoFia, a novel two-stage zero-shot fake news detection framework. First, we introduce Hierarchical Salience to quantify the importance of entities in the news content, and propose the SC-MMR algorithm to effectively select an informative and diverse set of keywords that serve as queries for retrieving up-to-date external evidence. Subsequently, a multi LLM interactive system, in which each agent assumes a distinct role, performs multi-view collaborative analysis and adversarial debate over the news text and its related information, and finally produces an interpretable and robust judgment. Comprehensive experiments on two public datasets demonstrate that ZoFia obviously outperforms existing zero-shot baselines and most of few-shot methods. Our codes will be open-sourced to facilitate related communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01187v1">Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely deployed for open-ended communication, yet most bias evaluations still rely on English, classification-style tasks. We introduce DebateBias-8K, a new multilingual, debate-style benchmark designed to reveal how narrative bias appears in realistic generative settings. Our dataset includes 8,400 structured debate prompts spanning four sensitive domains: women's rights, socioeconomic development, terrorism, and religion, across seven languages ranging from high-resource (English, Chinese) to low-resource (Swahili, Nigerian Pidgin). Using four flagship models (GPT-4o, Claude 3, DeepSeek, and LLaMA 3), we generate and automatically classify over 100,000 responses. Results show that all models reproduce entrenched stereotypes despite safety alignment: Arabs are overwhelmingly linked to terrorism and religion (>=95%), Africans to socioeconomic "backwardness" (up to <=77%), and Western groups are consistently framed as modern or progressive. Biases grow sharply in lower-resource languages, revealing that alignment trained primarily in English does not generalize globally. Our findings highlight a persistent divide in multilingual fairness: current alignment methods reduce explicit toxicity but fail to prevent biased outputs in open-ended contexts. We release our DebateBias-8K benchmark and analysis framework to support the next generation of multilingual bias evaluation and safer, culturally inclusive model alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01176v1">An Empirical Study of LLM-Based Code Clone Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in various software engineering tasks, such as code generation and debugging, because of their ability to translate between programming languages and natural languages. Existing studies have demonstrated the effectiveness of LLMs in code clone detection. However, two crucial issues remain unaddressed: the ability of LLMs to achieve comparable performance across different datasets and the consistency of LLMs' responses in code clone detection. To address these issues, we constructed seven code clone datasets and then evaluated five LLMs in four existing prompts with these datasets. The datasets were created by sampling code pairs using their Levenshtein ratio from two different code collections, CodeNet and BigCloneBench. Our evaluation revealed that although LLMs perform well in CodeNet-related datasets, with o3-mini achieving a 0.943 F1 score, their performance significantly decreased in BigCloneBench-related datasets. Most models achieved a high response consistency, with over 90\% of judgments remaining consistent across all five submissions. The fluctuations of the F1 score affected by inconsistency are also tiny; their variations are less than 0.03.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01166v1">MicroRemed: Benchmarking LLMs in Microservices Remediation</a></div>
    <div class="paper-meta">
      📅 2025-11-03
      | 💬 24 pages, 13 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) integrated with agent-based reasoning frameworks have recently shown strong potential for autonomous decision-making and system-level operations. One promising yet underexplored direction is microservice remediation, where the goal is to automatically recover faulty microservice systems. Existing approaches, however, still rely on human-crafted prompts from Site Reliability Engineers (SREs), with LLMs merely converting textual instructions into executable code. To advance research in this area, we introduce MicroRemed, the first benchmark for evaluating LLMs in end-to-end microservice remediation, where models must directly generate executable Ansible playbooks from diagnosis reports to restore system functionality. We further propose ThinkRemed, a multi-agent framework that emulates the reflective and perceptive reasoning of SREs. Experimental results show that MicroRemed presents substantial challenges to current LLMs, while ThinkRemed improves end-to-end remediation performance through iterative reasoning and system reflection. The benchmark is available at https://github.com/LLM4AIOps/MicroRemed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01144v1">AthenaBench: A Dynamic Benchmark for Evaluating LLMs in Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in natural language reasoning, yet their application to Cyber Threat Intelligence (CTI) remains limited. CTI analysis involves distilling large volumes of unstructured reports into actionable knowledge, a process where LLMs could substantially reduce analyst workload. CTIBench introduced a comprehensive benchmark for evaluating LLMs across multiple CTI tasks. In this work, we extend CTIBench by developing AthenaBench, an enhanced benchmark that includes an improved dataset creation pipeline, duplicate removal, refined evaluation metrics, and a new task focused on risk mitigation strategies. We evaluate twelve LLMs, including state-of-the-art proprietary models such as GPT-5 and Gemini-2.5 Pro, alongside seven open-source models from the LLaMA and Qwen families. While proprietary LLMs achieve stronger results overall, their performance remains subpar on reasoning-intensive tasks, such as threat actor attribution and risk mitigation, with open-source models trailing even further behind. These findings highlight fundamental limitations in the reasoning capabilities of current LLMs and underscore the need for models explicitly tailored to CTI workflows and automation.
    </details>
</div>
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00776v1">A Systematic Literature Review of Code Hallucinations in LLMs: Characterization, Mitigation Methods, Challenges, and Future Directions for Reliable AI</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Model hallucination is one of the most critical challenges faced by Large Language Models (LLMs), especially in high-stakes code intelligence tasks. As LLMs become increasingly integrated into software engineering tasks, understanding and mitigating hallucination in code becomes essential. In this survey, we provide a systematic review of hallucination phenomena in code-oriented LLMs from four key perspectives. First, we begin by surveying 60 papers to define hallucination in the context of code and summarize its primary causes, such as data noise, exposure bias, and insufficient semantic grounding, while also tracing recent trends in literature across natural language processing (NLP) and software engineering communities. Second, we review model hallucination surveys in a broader span and summarize representative hallucination mitigation strategies, such as knowledge-enhanced generation, constrained decoding, and post-editing. Third, we review approaches targeted for code intelligence and highlight code-specific challenges that aggravate hallucination, including syntax sensitivity, strict type systems, and dependence on external libraries. Meanwhile, we analyze how emerging code intelligence tasks, e.g., program analysis, symbolic execution, and unit testing, are utilized to detect and mitigate hallucinations. Fourth, we summarize current evaluation benchmarks, ranging from static metrics to dynamic checks, e.g., compilation and execution correctness, and emphasize the need for hallucination-oriented benchmarks.
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
