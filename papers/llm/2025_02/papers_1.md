# llm - 2025_02

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20383v1">Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Project website: http://vulnerable-ai-agents.github.io
    </div>
    <details class="paper-abstract">
      Recent advancements in Web AI agents have demonstrated remarkable capabilities in addressing complex web navigation tasks. However, emerging research shows that these agents exhibit greater vulnerability compared to standalone Large Language Models (LLMs), despite both being built upon the same safety-aligned models. This discrepancy is particularly concerning given the greater flexibility of Web AI Agent compared to standalone LLMs, which may expose them to a wider range of adversarial user inputs. To build a scaffold that addresses these concerns, this study investigates the underlying factors that contribute to the increased vulnerability of Web AI agents. Notably, this disparity stems from the multifaceted differences between Web AI agents and standalone LLMs, as well as the complex signals - nuances that simple evaluation metrics, such as success rate, often fail to capture. To tackle these challenges, we propose a component-level analysis and a more granular, systematic evaluation framework. Through this fine-grained investigation, we identify three critical factors that amplify the vulnerability of Web AI agents; (1) embedding user goals into the system prompt, (2) multi-step action generation, and (3) observational capabilities. Our findings highlights the pressing need to enhance security and robustness in AI agent design and provide actionable insights for targeted defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20356v1">Bridging the Creativity Understanding Gap: Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant limitations in understanding creative content, as demonstrated by Hessel et al. (2023)'s influential work on the New Yorker Cartoon Caption Contest (NYCCC). Their study exposed a substantial gap between LLMs and humans in humor comprehension, establishing that understanding and evaluating creative content is key challenge in AI development. We revisit this challenge by decomposing humor understanding into three components and systematically improve each: enhancing visual understanding through improved annotation, utilizing LLM-generated humor reasoning and explanations, and implementing targeted alignment with human preference data. Our refined approach achieves 82.4% accuracy in caption ranking, singificantly improving upon the previous 67% benchmark and matching the performance of world-renowned human experts in this domain. Notably, while attempts to mimic subgroup preferences through various persona prompts showed minimal impact, model finetuning with crowd preferences proved remarkably effective. These findings reveal that LLM limitations in creative judgment can be effectively addressed through focused alignment to specific subgroups and individuals. Lastly, we propose the position that achieving artificial general intelligence necessitates systematic collection of human preference data across creative domains. We advocate that just as human creativity is deeply influenced by individual and cultural preferences, training LLMs with diverse human preference data may be essential for developing true creative understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20284v1">Evaluating Human Trust in LLM-Based Planners: A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for planning tasks, offering unique capabilities not found in classical planners such as generating explanations and iterative refinement. However, trust--a critical factor in the adoption of planning systems--remains underexplored in the context of LLM-based planning tasks. This study bridges this gap by comparing human trust in LLM-based planners with classical planners through a user study in a Planning Domain Definition Language (PDDL) domain. Combining subjective measures, such as trust questionnaires, with objective metrics like evaluation accuracy, our findings reveal that correctness is the primary driver of trust and performance. Explanations provided by the LLM improved evaluation accuracy but had limited impact on trust, while plan refinement showed potential for increasing trust without significantly enhancing evaluation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v2">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02408v2">AI on My Shoulder: Supporting Emotional Labor in Front-Office Roles with an LLM-based Empathetic Coworker</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Client-Service Representatives (CSRs) are vital to organizations. Frequent interactions with disgruntled clients, however, disrupt their mental well-being. To help CSRs regulate their emotions while interacting with uncivil clients, we designed Care-Pilot, an LLM-powered assistant, and evaluated its efficacy, perception, and use. Our comparative analyses between 665 human and Care-Pilot-generated support messages highlight Care-Pilot's ability to adapt to and demonstrate empathy in various incivility incidents. Additionally, 143 CSRs assessed Care-Pilot's empathy as more sincere and actionable than human messages. Finally, we interviewed 20 CSRs who interacted with Care-Pilot in a simulation exercise. They reported that Care-Pilot helped them avoid negative thinking, recenter thoughts, and humanize clients; showing potential for bridging gaps in coworker support. Yet, they also noted deployment challenges and emphasized the indispensability of shared experiences. We discuss future designs and societal implications of AI-mediated emotional labor, underscoring empathy as a critical function for AI assistants for worker mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20258v1">LLM as a Broken Telephone: Iterative Generation Distorts Information</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      As large language models are increasingly responsible for online content, concerns arise about the impact of repeatedly processing their own outputs. Inspired by the "broken telephone" effect in chained human communication, this study investigates whether LLMs similarly distort information through iterative generation. Through translation-based experiments, we find that distortion accumulates over time, influenced by language choice and chain complexity. While degradation is inevitable, it can be mitigated through strategic prompting techniques. These findings contribute to discussions on the long-term effects of AI-mediated information propagation, raising important questions about the reliability of LLM-generated content in iterative workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20238v1">FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Many challenging reasoning tasks require not just rapid, intuitive responses, but a more deliberate, multi-step approach. Recent progress in large language models (LLMs) highlights an important shift from the "System 1" way of quick reactions to the "System 2" style of reflection-and-correction problem solving. However, current benchmarks heavily rely on the final-answer accuracy, leaving much of a model's intermediate reasoning steps unexamined. This fails to assess the model's ability to reflect and rectify mistakes within the reasoning process. To bridge this gap, we introduce FINEREASON, a logic-puzzle benchmark for fine-grained evaluation of LLMs' reasoning capabilities. Each puzzle can be decomposed into atomic steps, making it ideal for rigorous validation of intermediate correctness. Building on this, we introduce two tasks: state checking, and state transition, for a comprehensive evaluation of how models assess the current situation and plan the next move. To support broader research, we also provide a puzzle training set aimed at enhancing performance on general mathematical tasks. We show that models trained on our state checking and transition data demonstrate gains in math reasoning by up to 5.1% on GSM8K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17732v2">CheckMate: LLM-Powered Approximate Intermittent Computing</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Accepted in SenSys 2025
    </div>
    <details class="paper-abstract">
      Batteryless IoT systems face energy constraints exacerbated by checkpointing overhead. Approximate computing offers solutions but demands manual expertise, limiting scalability. This paper presents CheckMate, an automated framework leveraging LLMs for context-aware code approximations. CheckMate integrates validation of LLM-generated approximations to ensure correct execution and employs Bayesian optimization to fine-tune approximation parameters autonomously, eliminating the need for developer input. Tested across six IoT applications, it reduces power cycles by up to 60% with an accuracy loss of just 8%, outperforming semi-automated tools like ACCEPT in speedup and accuracy. CheckMate's results establish it as a robust, user-friendly tool and a foundational step toward automated approximation frameworks for intermittent computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20175v1">An Extensive Evaluation of PDDL Capabilities in off-the-shelf LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      In recent advancements, large language models (LLMs) have exhibited proficiency in code generation and chain-of-thought reasoning, laying the groundwork for tackling automatic formal planning tasks. This study evaluates the potential of LLMs to understand and generate Planning Domain Definition Language (PDDL), an essential representation in artificial intelligence planning. We conduct an extensive analysis across 20 distinct models spanning 7 major LLM families, both commercial and open-source. Our comprehensive evaluation sheds light on the zero-shot LLM capabilities of parsing, generating, and reasoning with PDDL. Our findings indicate that while some models demonstrate notable effectiveness in handling PDDL, others pose limitations in more complex scenarios requiring nuanced planning knowledge. These results highlight the promise and current limitations of LLMs in formal planning tasks, offering insights into their application and guiding future efforts in AI-driven planning paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20140v1">Telephone Surveys Meet Conversational AI: Evaluating a LLM-Based Telephone Survey System at Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Telephone surveys remain a valuable tool for gathering insights but typically require substantial resources in training and coordinating human interviewers. This work presents an AI-driven telephone survey system integrating text-to-speech (TTS), a large language model (LLM), and speech-to-text (STT) that mimics the versatility of human-led interviews on scale. We tested the system across two populations, a pilot study in the United States (n = 75) and a large-scale deployment in Peru (n = 2,739), inviting participants via web-based links and contacting them via direct phone calls. The AI agent successfully administered open-ended and closed-ended questions, handled basic clarifications, and dynamically navigated branching logic, allowing fast large-scale survey deployment without interviewer recruitment or training. Our findings demonstrate that while the AI system's probing for qualitative depth was more limited than human interviewers, overall data quality approached human-led standards for structured items. This study represents one of the first successful large-scale deployments of an LLM-based telephone interviewer in a real-world survey context. The AI-powered telephone survey system has the potential for expanding scalable, consistent data collecting across market research, social science, and public opinion studies, thus improving operational efficiency while maintaining appropriate data quality for research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11807v6">How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Accepted to ICLR 2025; 11 pages of main text; 26 pages of appendices; Included models: GPT-3.5-{0613, 1106, 0125}, GPT-4-0125, GPT-4o-0806, Gemini-{1.0, 1.5)-Pro, LLaMA-3.1-{7, 70, 405}B, Mixtral-8x{7, 22}B, Qwen-2-72B
    </div>
    <details class="paper-abstract">
      Decision-making is a complex process requiring diverse abilities, making it an excellent framework for evaluating Large Language Models (LLMs). Researchers have examined LLMs' decision-making through the lens of Game Theory. However, existing evaluation mainly focus on two-player scenarios where an LLM competes against another. Additionally, previous benchmarks suffer from test set leakage due to their static design. We introduce GAMA($\gamma$)-Bench, a new framework for evaluating LLMs' Gaming Ability in Multi-Agent environments. It includes eight classical game theory scenarios and a dynamic scoring scheme specially designed to quantitatively assess LLMs' performance. $\gamma$-Bench allows flexible game settings and adapts the scoring system to different game parameters, enabling comprehensive evaluation of robustness, generalizability, and strategies for improvement. Our results indicate that GPT-3.5 demonstrates strong robustness but limited generalizability, which can be enhanced using methods like Chain-of-Thought. We also evaluate 13 LLMs from 6 model families, including GPT-3.5, GPT-4, Gemini, LLaMA-3.1, Mixtral, and Qwen-2. Gemini-1.5-Pro outperforms others, scoring of $69.8$ out of $100$, followed by LLaMA-3.1-70B ($65.9$) and Mixtral-8x22B ($62.4$). Our code and experimental results are publicly available at https://github.com/CUHK-ARISE/GAMABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20082v1">LongRoPE2: Near-Lossless LLM Context Window Scaling</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      LongRoPE2 is a novel approach that extends the effective context window of pre-trained large language models (LLMs) to the target length, while preserving the performance on the original shorter context window. This is achieved by three contributions: (1) a hypothesis that insufficient training in higher RoPE dimensions contributes to the persistent out-of-distribution (OOD) issues observed in existing methods; (2) an effective RoPE rescaling algorithm that adopts evolutionary search guided by "needle-driven" perplexity to address the insufficient training problem; (3) a mixed context window training approach that fine-tunes model weights to adopt rescaled RoPE for long-context sequences while preserving the short-context performance with the original RoPE. Extensive experiments on LLaMA3-8B and Phi3-mini-3.8B across various benchmarks validate the hypothesis and demonstrate the effectiveness of LongRoPE2. Remarkably, LongRoPE2 extends LLaMA3-8B to achieve a 128K effective context length while retaining over 98.5% of short-context performance, using only 10B tokens -- 80x fewer than Meta's approach, which fails to reach the target effective context length. Code will be available at https://github.com/microsoft/LongRoPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06153v3">AgentSquare: Automatic LLM Agent Search in Modular Design Space</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to a rapid growth of agentic systems capable of handling a wide range of complex tasks. However, current research largely relies on manual, task-specific design, limiting their adaptability to novel tasks. In this paper, we introduce a new research problem: Modularized LLM Agent Search (MoLAS). We propose a modular design space that abstracts existing LLM agent designs into four fundamental modules with uniform IO interface: Planning, Reasoning, Tool Use, and Memory. Building on this design space, we present a novel LLM agent search framework called AgentSquare, which introduces two core mechanisms, i.e., module evolution and recombination, to efficiently search for optimized LLM agents. To further accelerate the process, we design a performance predictor that uses in-context surrogate models to skip unpromising agent designs. Extensive experiments across six benchmarks, covering the diverse scenarios of web, embodied, tool use and game applications, show that AgentSquare substantially outperforms hand-crafted agents, achieving an average performance gain of 17.2% against best-known human designs. Moreover, AgentSquare can generate interpretable design insights, enabling a deeper understanding of agentic architecture and its impact on task performance. We believe that the modular design space and AgentSquare search framework offer a platform for fully exploiting the potential of prior successful designs and consolidating the collective efforts of research community. Code repo is available at https://github.com/tsinghua-fib-lab/AgentSquare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06899v2">LongSafety: Enhance Safety for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Recent advancements in model architectures and length extrapolation techniques have significantly extended the context length of large language models (LLMs), paving the way for their application in increasingly complex tasks. However, despite the growing capabilities of long-context LLMs, the safety issues in long-context scenarios remain underexplored. While safety alignment in short context has been widely studied, the safety concerns of long-context LLMs have not been adequately addressed. In this work, we introduce \textbf{LongSafety}, a comprehensive safety alignment dataset for long-context LLMs, containing 10 tasks and 17k samples, with an average length of 40.9k tokens. Our experiments demonstrate that training with LongSafety can enhance long-context safety performance while enhancing short-context safety and preserving general capabilities. Furthermore, we demonstrate that long-context safety does not equal long-context alignment with short-context safety data and LongSafety has generalizing capabilities in context length and long-context safety scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04671v2">CUIfy the XR: An Open-Source Package to Embed LLM-powered Conversational Agents in XR</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 7th IEEE International Conference on Artificial Intelligence & eXtended and Virtual Reality (IEEE AIxVR 2025)
    </div>
    <details class="paper-abstract">
      Recent developments in computer graphics, machine learning, and sensor technologies enable numerous opportunities for extended reality (XR) setups for everyday life, from skills training to entertainment. With large corporations offering affordable consumer-grade head-mounted displays (HMDs), XR will likely become pervasive, and HMDs will develop as personal devices like smartphones and tablets. However, having intelligent spaces and naturalistic interactions in XR is as important as technological advances so that users grow their engagement in virtual and augmented spaces. To this end, large language model (LLM)--powered non-player characters (NPCs) with speech-to-text (STT) and text-to-speech (TTS) models bring significant advantages over conventional or pre-scripted NPCs for facilitating more natural conversational user interfaces (CUIs) in XR. This paper provides the community with an open-source, customizable, extendable, and privacy-aware Unity package, CUIfy, that facilitates speech-based NPC-user interaction with widely used LLMs, STT, and TTS models. Our package also supports multiple LLM-powered NPCs per environment and minimizes latency between different computational models through streaming to achieve usable interactions between users and NPCs. We publish our source code in the following repository: https://gitlab.lrz.de/hctl/cuify
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01928v2">MALT: Improving Reasoning with Multi-Agent LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce answers with a single chain-of-thought, which restricts their ability to explore reasoning paths or self-correct flawed outputs in complex tasks. In this paper, we introduce MALT (Multi-Agent LLM Training), a novel post-training strategy that divides the reasoning process into generation, verification, and refinement steps using a sequential pipeline of heterogeneous agents. During data generation, each agent is repeatedly sampled to form a multi-agent search tree, where final outputs are graded against ground-truth data. We then apply value iteration to propagate reward signals back to each role-conditioned model, automatically producing multi-agent post-training data without human or teacher-model supervision. Our off-policy approach allows each agent to specialize by learning from correct and incorrect trajectories, ultimately improving the end-to-end reasoning chain. On MATH, GSM8K, and CSQA, MALT surpasses the same baseline LLM with a relative improvement of 15.66%, 7.42%, and 9.40% respectively, making it an important advance towards multi-agent cooperative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13461v2">Progressive Mixed-Precision Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      In spite of the great potential of large language models (LLMs) across various tasks, their deployment on resource-constrained devices remains challenging due to their excessive computational and memory demands. Quantization has emerged as an effective solution by storing weights in reduced precision. However, utilizing low precisions (i.e.~2/3-bit) to substantially alleviate the memory-boundedness of LLM decoding, still suffers from prohibitive performance drop. In this work, we argue that existing approaches fail to explore the diversity in computational patterns, redundancy, and sensitivity to approximations of the different phases of LLM inference, resorting to a uniform quantization policy throughout. Instead, we propose a novel phase-aware method that selectively allocates precision during different phases of LLM inference, achieving both strong context extraction during prefill and efficient memory bandwidth utilization during decoding. To further address the memory-boundedness of the decoding phase, we introduce Progressive Mixed-Precision Decoding (PMPD), a technique that enables the gradual lowering of precision deeper in the generated sequence, together with a spectrum of precision-switching schedulers that dynamically drive the precision-lowering decisions in either task-adaptive or prompt-adaptive manner. Extensive evaluation across diverse language tasks shows that when targeting Nvidia GPUs, PMPD achieves 1.4$-$12.2$\times$ speedup in matrix-vector multiplications over fp16 models, while when targeting an LLM-optimized NPU, our approach delivers a throughput gain of 3.8$-$8.0$\times$ over fp16 models and up to 1.54$\times$ over uniform quantization approaches while preserving the output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19981v1">The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Autoregressive large language models (LLMs) exhibit impressive performance across various tasks but struggle with simple arithmetic, such as addition of two or more operands. We show that this struggle arises from LLMs' use of a simple one-digit lookahead heuristic, which works fairly well (but not perfect) for two-operand addition but fails in multi-operand cases, where the carry-over logic is more complex. Our probing experiments and digit-wise accuracy evaluation show that LLMs fail precisely where a one-digit lookahead is insufficient to account for cascading carries. We analyze the impact of tokenization strategies on arithmetic performance and show that all investigated models, regardless of tokenization, are inherently limited in the addition of multiple operands due to their reliance on a one-digit lookahead heuristic. Our findings reveal fundamental limitations that prevent LLMs from generalizing to more complex numerical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19965v1">Deterministic or probabilistic? The psychology of LLMs as random number generators</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 31 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed text generation through inherently probabilistic context-aware mechanisms, mimicking human natural language. In this paper, we systematically investigate the performance of various LLMs when generating random numbers, considering diverse configurations such as different model architectures, numerical ranges, temperature, and prompt languages. Our results reveal that, despite their stochastic transformers-based architecture, these models often exhibit deterministic responses when prompted for random numerical outputs. In particular, we find significant differences when changing the model, as well as the prompt language, attributing this phenomenon to biases deeply embedded within the training data. Models such as DeepSeek-R1 can shed some light on the internal reasoning process of LLMs, despite arriving to similar results. These biases induce predictable patterns that undermine genuine randomness, as LLMs are nothing but reproducing our own human cognitive biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v2">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19915v1">LLM-driven Effective Knowledge Tracing by Integrating Dual-channel Difficulty</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Knowledge Tracing (KT) is a fundamental technology in intelligent tutoring systems used to simulate changes in students' knowledge state during learning, track personalized knowledge mastery, and predict performance. However, current KT models face three major challenges: (1) When encountering new questions, models face cold-start problems due to sparse interaction records, making precise modeling difficult; (2) Traditional models only use historical interaction records for student personalization modeling, unable to accurately track individual mastery levels, resulting in unclear personalized modeling; (3) The decision-making process is opaque to educators, making it challenging for them to understand model judgments. To address these challenges, we propose a novel Dual-channel Difficulty-aware Knowledge Tracing (DDKT) framework that utilizes Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) for subjective difficulty assessment, while integrating difficulty bias-aware algorithms and student mastery algorithms for precise difficulty measurement. Our framework introduces three key innovations: (1) Difficulty Balance Perception Sequence (DBPS) - students' subjective perceptions combined with objective difficulty, measuring gaps between LLM-assessed difficulty, mathematical-statistical difficulty, and students' subjective perceived difficulty through attention mechanisms; (2) Difficulty Mastery Ratio (DMR) - precise modeling of student mastery levels through different difficulty zones; (3) Knowledge State Update Mechanism - implementing personalized knowledge acquisition through gated networks and updating student knowledge state. Experimental results on two real datasets show our method consistently outperforms nine baseline models, improving AUC metrics by 2% to 10% while effectively addressing cold-start problems and enhancing model interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19913v1">SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Data and pipeline parallelism are ubiquitous for training of Large Language Models (LLM) on distributed nodes. Driven by the need for cost-effective training, recent work explores efficient communication arrangement for end to end training. Motivated by LLM's resistance to layer skipping and layer reordering, in this paper, we explore stage (several consecutive layers) skipping in pipeline training, and challenge the conventional practice of sequential pipeline execution. We derive convergence and throughput constraints (guidelines) for pipelining with skipping and swapping pipeline stages. Based on these constraints, we propose SkipPipe, the first partial pipeline framework to reduce the end-to-end training time for LLMs while preserving the convergence. The core of SkipPipe is a path scheduling algorithm that optimizes the paths for individual microbatches and reduces idle time (due to microbatch collisions) on the distributed nodes, complying with the given stage skipping ratio. We extensively evaluate SkipPipe on LLaMa models from 500M to 8B parameters on up to 20 nodes. Our results show that SkipPipe reduces training iteration time by up to $55\%$ compared to full pipeline. Our partial pipeline training also improves resistance to layer omission during inference, experiencing a drop in perplexity of only $7\%$ when running only half the model. Our code is available at https://github.com/gensyn-ai/skippipe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19907v1">Order Doesn't Matter, But Reasoning Does: Training LLMs with Order-Centric Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Logical reasoning is essential for large language models (LLMs) to ensure accurate and coherent inference. However, LLMs struggle with reasoning order variations and fail to generalize across logically equivalent transformations. LLMs often rely on fixed sequential patterns rather than true logical understanding. To address this issue, we introduce an order-centric data augmentation framework based on commutativity in logical reasoning. We first randomly shuffle independent premises to introduce condition order augmentation. For reasoning steps, we construct a directed acyclic graph (DAG) to model dependencies between steps, which allows us to identify valid reorderings of steps while preserving logical correctness. By leveraging order-centric augmentations, models can develop a more flexible and generalized reasoning process. Finally, we conduct extensive experiments across multiple logical reasoning benchmarks, demonstrating that our method significantly enhances LLMs' reasoning performance and adaptability to diverse logical structures. We release our codes and augmented data in https://anonymous.4open.science/r/Order-Centric-Data-Augmentation-822C/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13952v2">The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility?</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Recent years have witnessed extensive efforts to enhance Large Language Models (LLMs) across various domains, alongside growing attention to their ethical implications. However, a critical challenge remains largely overlooked: LLMs must balance between rejecting harmful requests for safety and accommodating legitimate ones for utility. This paper presents a Direct Preference Optimization (DPO) based alignment framework that achieves better overall performance by addressing this ethical-utility trade-off, using chemical domain applications as a proof-of-concept. Our alignment pipeline starts with a GPT-assisted three-phase data generation scheme, in which we create LibraChemQA, a chemical question-answering dataset comprising 31.6k triplet instances. By incorporating an innovative balanced seed in the data generation process, our framework systematically considers both legitimate and illegitimate requests. The framework also introduces a rephrasing mechanism for efficient data augmentation that enhances the model's chemical comprehension. We further develop a novel hybrid evaluation scheme with LLM judges for precise assessment of both safety and utility. Experimental results demonstrate our model's substantial improvements in overall performance where both safety and utility are considered - the resulting model outperforms leading LLMs including Claude-3, GPT-4o, and LLaMA-3 by margins of 13.44%, 7.16%, and 7.10% respectively on our released benchmark. At the end of this paper, we analyze experimental results obtained from testing DeepSeek-R1 on our benchmark and reveal the critical ethical concerns raised by this highly acclaimed model. We highlight that the long Chain-of-Thought (CoT) reasoning process employed by DeepSeek-R1, as well as other LLMs distilled from it, introduces significant ethical vulnerabilities when exposed to users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23769v2">The Potential of LLMs in Medical Education: Generating Questions and Answers for Qualification Exams</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      In this work, we leverage LLMs to produce medical qualification exam questions and the corresponding answers through few-shot prompts, investigating in-depth how LLMs meet the requirements in terms of coherence, evidence of statement, factual consistency, and professionalism etc. Utilizing a multicenter bidirectional anonymized database with respect to comorbid chronic diseases, named Elderly Comorbidity Medical Database (CECMed), we tasked LLMs with generating open-ended questions and answers based on a subset of sampled admission reports. For CECMed, the retrospective cohort includes patients enrolled from January 2010 to January 2022 while the prospective cohort from January 2023 to November 2023, with participants sourced from selected tertiary and community hospitals across the southern, northern, and central regions of China. A total of 8 widely used LLMs were used, including ERNIE 4, ChatGLM 4, Doubao, Hunyuan, Spark 4, Qwen, Conventional medical education requires sophisticated clinicians to formulate questions and answers based on prototypes from EHRs, which is heuristic and time-consuming. We found that mainstream LLMs could generate questions and answers with real-world EHRs at levels close to clinicians. Although current LLMs performed dissatisfactory in some aspects, medical students, interns and residents could reasonably make use of LLMs to facilitate understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19271v2">AutoPureData: Automated Filtering of Undesirable Web Data to Update LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Final version
    </div>
    <details class="paper-abstract">
      Up-to-date and reliable language models are consistently sought after and are essential in various applications. Typically, models are trained on a fixed dataset and then deployed globally. However, the knowledge of the models becomes outdated. Enabling automatic updation of AI knowledge using web data involves significant concerns regarding the model's safety and quality due to a threat from unsafe and undesirable text across the web. The purity of new data was essential for updating knowledge of language models to maintain their reliability. This paper proposes AutoPureData, a system that automatically collects and purifies web data. The system loaded a sample of web data. Utilizing existing trusted AI models, it successfully eliminated unsafe text with an accuracy of 97% and undesirable text with an accuracy of 86%, demonstrating the system's effectiveness in purifying the data. The system ensures that only meaningful and safe text can be used to update LLM knowledge. The pure text was then optimized and stored in a vector database for future querying. It was found that LLM can fetch new data from the vector DB. The LLM writes the RAG query in English, even if the user's query is in another language, proving that the system can perform cross-lingual retrieval. This paper proposes a method to maintain the accuracy and relevance of up-to-date language models by ensuring that only purified data was used to update LLM knowledge. This work contributes to updating knowledge of chatbots using meaningful and safe text, enhancing their utility across various industries, and potentially reducing the risks associated with outputs caused by unsafe or impure data. Code is available at github.com/Pro-GenAI/AutoPureData.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15308v2">LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Content</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 LLMs, Multilingual, Language Diversity, Large Language Models, Social Media, News Media, Specialized LLMs, Fact-checking, Media Analysis, Arabic, Hindi, English
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success as general-purpose task solvers across various fields. However, their capabilities remain limited when addressing domain-specific problems, particularly in downstream NLP tasks. Research has shown that models fine-tuned on instruction-based downstream NLP datasets outperform those that are not fine-tuned. While most efforts in this area have primarily focused on resource-rich languages like English and broad domains, little attention has been given to multilingual settings and specific domains. To address this gap, this study focuses on developing a specialized LLM, LlamaLens, for analyzing news and social media content in a multilingual context. To the best of our knowledge, this is the first attempt to tackle both domain specificity and multilinguality, with a particular focus on news and social media. Our experimental setup includes 18 tasks, represented by 52 datasets covering Arabic, English, and Hindi. We demonstrate that LlamaLens outperforms the current state-of-the-art (SOTA) on 23 testing sets, and achieves comparable performance on 8 sets. We make the models and resources publicly available for the research community (https://huggingface.co/collections/QCRI/llamalens-672f7e0604a0498c6a2f0fe9).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19820v1">Foot-In-The-Door: A Multi-turn Jailbreak for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions.Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions.The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16235v2">Dynamic Parallel Tree Search for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Tree of Thoughts (ToT) enhances Large Language Model (LLM) reasoning by structuring problem-solving as a spanning tree. However, recent methods focus on search accuracy while overlooking computational efficiency. The challenges of accelerating the ToT lie in the frequent switching of reasoning focus, and the redundant exploration of suboptimal solutions. To alleviate this dilemma, we propose Dynamic Parallel Tree Search (DPTS), a novel parallelism framework that aims to dynamically optimize the reasoning path in inference. It includes the Parallelism Streamline in the generation phase to build up a flexible and adaptive parallelism with arbitrary paths by fine-grained cache management and alignment. Meanwhile, the Search and Transition Mechanism filters potential candidates to dynamically maintain the reasoning focus on more possible solutions and have less redundancy. Experiments on Qwen-2.5 and Llama-3 with Math500 and GSM8K datasets show that DPTS significantly improves efficiency by 2-4x on average while maintaining or even surpassing existing reasoning algorithms in accuracy, making ToT-based reasoning more scalable and computationally efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17529v2">ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      This paper proposes a novel method for multi-lane convoy formation control that uses large language models (LLMs) to tackle coordination challenges in dynamic highway environments. Each connected and autonomous vehicle in the convoy uses a knowledge-driven approach to make real-time adaptive decisions based on various scenarios. Our method enables vehicles to dynamically perform tasks, including obstacle avoidance, convoy joining/leaving, and escort formation switching, all while maintaining the overall convoy structure. We design a Interlaced formation control strategy based on locally dynamic distributed graphs, ensuring the convoy remains stable and flexible. We conduct extensive experiments in the SUMO simulation platform across multiple traffic scenarios, and the results demonstrate that the proposed method is effective, robust, and adaptable to dynamic environments. The code is available at: https://github.com/chuduanfeng/ConvoyLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v3">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13834v3">Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Published as a conference paper at ICLR 2025. Code is available at https://github.com/Lizn-zn/NeqLIPS/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19735v1">R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Despite recent breakthroughs in reasoning-enhanced large language models (LLMs) like DeepSeek-R1, incorporating inference-time reasoning into machine translation (MT), where human translators naturally employ structured, multi-layered reasoning chain-of-thoughts (CoTs), is yet underexplored. Existing methods either design a fixed CoT tailored for a specific MT sub-task (e.g., literature translation), or rely on synthesizing CoTs unaligned with humans and supervised fine-tuning (SFT) prone to catastrophic forgetting, limiting their adaptability to diverse translation scenarios. This paper introduces R1-Translator (R1-T1), a novel framework to achieve inference-time reasoning for general MT via reinforcement learning (RL) with human-aligned CoTs comprising six common patterns. Our approach pioneers three innovations: (1) extending reasoning-based translation beyond MT sub-tasks to six languages and diverse tasks (e.g., legal/medical domain adaptation, idiom resolution); (2) formalizing six expert-curated CoT templates that mirror hybrid human strategies like context-aware paraphrasing and back translation; and (3) enabling self-evolving CoT discovery and anti-forgetting adaptation through RL with KL-constrained rewards. Experimental results indicate a steady translation performance improvement in 21 languages and 80 translation directions on Flores-101 test set, especially on the 15 languages unseen from training, with its general multilingual abilities preserved compared with plain SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19731v1">Preference Learning Unlocks LLMs' Psycho-Counseling Skills</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Applying large language models (LLMs) to assist in psycho-counseling is an emerging and meaningful approach, driven by the significant gap between patient needs and the availability of mental health support. However, current LLMs struggle to consistently provide effective responses to client speeches, largely due to the lack of supervision from high-quality real psycho-counseling data, whose content is typically inaccessible due to client privacy concerns. Furthermore, the quality of therapists' responses in available sessions can vary significantly based on their professional training and experience. Assessing the quality of therapists' responses remains an open challenge. In this work, we address these challenges by first proposing a set of professional and comprehensive principles to evaluate therapists' responses to client speeches. Using these principles, we create a preference dataset, PsychoCounsel-Preference, which contains 36k high-quality preference comparison pairs. This dataset aligns with the preferences of professional psychotherapists, providing a robust foundation for evaluating and improving LLMs in psycho-counseling. Experiments on reward modeling and preference learning demonstrate that PsychoCounsel-Preference is an excellent resource for LLMs to acquire essential skills for responding to clients in a counseling session. Our best-aligned model, PsychoCounsel-Llama3-8B, achieves an impressive win rate of 87% against GPT-4o. We release PsychoCounsel-Preference, PsychoCounsel-Llama3-8B and the reward model PsychoCounsel Llama3-8B-Reward to facilitate the research of psycho-counseling with LLMs at: https://hf.co/Psychotherapy-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19721v1">Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to perpetuate stereotypes and exhibit biases. Various strategies have been proposed to mitigate potential harms that may result from these biases, but most work studies biases in LLMs as a black-box problem without considering how concepts are represented within the model. We adapt techniques from representation engineering to study how the concept of "gender" is represented within LLMs. We introduce a new method that extracts concept representations via probability weighting without labeled data and efficiently selects a steering vector for measuring and manipulating the model's representation. We also present a projection-based method that enables precise steering of model predictions and demonstrate its effectiveness in mitigating gender bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02926v4">Demystifying RCE Vulnerabilities in LLM-Integrated Apps</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      LLMs show promise in transforming software development, with a growing interest in integrating them into more intelligent apps. Frameworks like LangChain aid LLM-integrated app development, offering code execution utility/APIs for custom actions. However, these capabilities theoretically introduce Remote Code Execution (RCE) vulnerabilities, enabling remote code execution through prompt injections. No prior research systematically investigates these frameworks' RCE vulnerabilities or their impact on applications and exploitation consequences. Therefore, there is a huge research gap in this field. In this study, we propose LLMSmith to detect, validate and exploit the RCE vulnerabilities in LLM-integrated frameworks and apps. To achieve this goal, we develop two novel techniques, including 1) a lightweight static analysis to examine LLM integration mechanisms, and construct call chains to identify RCE vulnerabilities in frameworks; 2) a systematical prompt-based exploitation method to verify and exploit the found vulnerabilities in LLM-integrated apps. This technique involves various strategies to control LLM outputs, trigger RCE vulnerabilities and launch subsequent attacks. Our research has uncovered a total of 20 vulnerabilities in 11 LLM-integrated frameworks, comprising 19 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. Of these, 17 have been confirmed by the framework developers, with 11 vulnerabilities being assigned CVE IDs. For the 51 apps potentially affected by RCE, we successfully executed attacks on 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. Furthermore, we conduct a comprehensive analysis of these vulnerabilities and construct practical attacks to demonstrate the hazards in reality. Last, we propose several mitigation measures for both framework and app developers to counteract such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20681v2">No Free Lunch Theorem for Privacy-Preserving LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Individuals and businesses have been significantly benefited by Large Language Models (LLMs) including PaLM, Gemini and ChatGPT in various ways. For example, LLMs enhance productivity, reduce costs, and enable us to focus on more valuable tasks. Furthermore, LLMs possess the capacity to sift through extensive datasets, uncover underlying patterns, and furnish critical insights that propel the frontiers of technology and science. However, LLMs also pose privacy concerns. Users' interactions with LLMs may expose their sensitive personal or company information. A lack of robust privacy safeguards and legal frameworks could permit the unwarranted intrusion or improper handling of individual data, thereby risking infringements of privacy and the theft of personal identities. To ensure privacy, it is essential to minimize the dependency between shared prompts and private information. Various randomization approaches have been proposed to protect prompts' privacy, but they may incur utility loss compared to unprotected LLMs prompting. Therefore, it is essential to evaluate the balance between the risk of privacy leakage and loss of utility when conducting effective protection mechanisms. The current study develops a framework for inferring privacy-protected Large Language Models (LLMs) and lays down a solid theoretical basis for examining the interplay between privacy preservation and utility. The core insight is encapsulated within a theorem that is called as the NFL (abbreviation of the word No-Free-Lunch) Theorem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08904v3">MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Recent methodologies utilizing synthetic datasets have aimed to address inconsistent hallucinations in large language models (LLMs); however,these approaches are primarily tailored to specific tasks, limiting their generalizability. Inspired by the strong performance of code-trained models in logic-intensive domains, we propose a novel framework that leverages event-based text to generate corresponding code and employs cyclic training to transfer the logical consistency of code to natural language effectively. Our method significantly reduces inconsistent hallucinations across three leading LLMs and two categories of natural language tasks while maintaining overall performance. This framework effectively alleviates hallucinations without necessitating adaptation to downstream tasks, demonstrating generality and providing new perspectives to tackle the challenge of inconsistent hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19669v1">Investigating Neurons and Heads in Transformer-based LLMs for Typographical Errors</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 14 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      This paper investigates how LLMs encode inputs with typos. We hypothesize that specific neurons and attention heads recognize typos and fix them internally using local and global contexts. We introduce a method to identify typo neurons and typo heads that work actively when inputs contain typos. Our experimental results suggest the following: 1) LLMs can fix typos with local contexts when the typo neurons in either the early or late layers are activated, even if those in the other are not. 2) Typo neurons in the middle layers are responsible for the core of typo-fixing with global contexts. 3) Typo heads fix typos by widely considering the context not focusing on specific tokens. 4) Typo neurons and typo heads work not only for typo-fixing but also for understanding general contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19662v1">HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Quantization is critical for realizing efficient inference of LLMs. Traditional quantization methods are hardware-agnostic, limited to bit-width constraints, and lacking circuit-level insights, such as timing and energy characteristics of Multiply-Accumulate (MAC) units. We introduce HALO, a versatile framework that adapts to various hardware through a Hardware-Aware Post-Training Quantization (PTQ) approach. By leveraging MAC unit properties, HALO minimizes critical-path delays and enables dynamic frequency scaling. Deployed on LLM accelerators like TPUs and GPUs, HALO achieves on average 270% performance gains and 51% energy savings, all with minimal accuracy drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19413v1">Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Paywalls, licenses and copyright rules often restrict the broad dissemination and reuse of scientific knowledge. We take the position that it is both legally and technically feasible to extract the scientific knowledge in scholarly texts. Current methods, like text embeddings, fail to reliably preserve factual content, and simple paraphrasing may not be legally sound. We urge the community to adopt a new idea: convert scholarly documents into Knowledge Units using LLMs. These units use structured data capturing entities, attributes and relationships without stylistic content. We provide evidence that Knowledge Units: (1) form a legally defensible framework for sharing knowledge from copyrighted research texts, based on legal analyses of German copyright law and U.S. Fair Use doctrine, and (2) preserve most (~95%) factual knowledge from original text, measured by MCQ performance on facts from the original copyrighted text across four research domains. Freeing scientific knowledge from copyright promises transformative benefits for scientific research and education by allowing language models to reuse important facts from copyrighted text. To support this, we share open-source tools for converting research documents into Knowledge Units. Overall, our work posits the feasibility of democratizing access to scientific knowledge while respecting copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19411v1">Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Project Repo: https://github.com/dayuyang1999/Awesome-Code-Reasoning
    </div>
    <details class="paper-abstract">
      In large language models (LLMs), code and reasoning reinforce each other: code offers an abstract, modular, and logic-driven structure that supports reasoning, while reasoning translates high-level goals into smaller, executable steps that drive more advanced code intelligence. In this study, we examine how code serves as a structured medium for enhancing reasoning: it provides verifiable execution paths, enforces logical decomposition, and enables runtime validation. We also explore how improvements in reasoning have transformed code intelligence from basic completion to advanced capabilities, enabling models to address complex software engineering tasks through planning and debugging. Finally, we identify key challenges and propose future research directions to strengthen this synergy, ultimately improving LLM's performance in both areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19410v1">Less or More: Towards Glanceable Explanations for LLM Recommendations Using Ultra-Small Devices</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable potential in recommending everyday actions as personal AI assistants, while Explainable AI (XAI) techniques are being increasingly utilized to help users understand why a recommendation is given. Personal AI assistants today are often located on ultra-small devices such as smartwatches, which have limited screen space. The verbosity of LLM-generated explanations, however, makes it challenging to deliver glanceable LLM explanations on such ultra-small devices. To address this, we explored 1) spatially structuring an LLM's explanation text using defined contextual components during prompting and 2) presenting temporally adaptive explanations to users based on confidence levels. We conducted a user study to understand how these approaches impacted user experiences when interacting with LLM recommendations and explanations on ultra-small devices. The results showed that structured explanations reduced users' time to action and cognitive load when reading an explanation. Always-on structured explanations increased users' acceptance of AI recommendations. However, users were less satisfied with structured explanations compared to unstructured ones due to their lack of sufficient, readable details. Additionally, adaptively presenting structured explanations was less effective at improving user perceptions of the AI compared to the always-on structured explanations. Together with users' interview feedback, the results led to design implications to be mindful of when personalizing the content and timing of LLM explanations that are displayed on ultra-small devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19400v1">TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed long-form videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15823v2">InductionBench: LLMs Fail in the Simplest Complexity Class</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 24 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable improvements in reasoning and many existing benchmarks have been addressed by models such as o1 and o3 either fully or partially. However, a majority of these benchmarks emphasize deductive reasoning, including mathematical and coding tasks in which rules such as mathematical axioms or programming syntax are clearly defined, based on which LLMs can plan and apply these rules to arrive at a solution. In contrast, inductive reasoning, where one infers the underlying rules from observed data, remains less explored. Such inductive processes lie at the heart of scientific discovery, as they enable researchers to extract general principles from empirical observations. To assess whether LLMs possess this capacity, we introduce InductionBench, a new benchmark designed to evaluate the inductive reasoning ability of LLMs. Our experimental findings reveal that even the most advanced models available struggle to master the simplest complexity classes within the subregular hierarchy of functions, highlighting a notable deficiency in current LLMs' inductive reasoning capabilities. Coda and data are available https://github.com/Wenyueh/inductive_reasoning_benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04318v2">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Under review at ICLR
    </div>
    <details class="paper-abstract">
      This paper introduces the counter-intuitive generalization results of overfitting pre-trained large language models (LLMs) on very small datasets. In the setting of open-ended text generation, it is well-documented that LLMs tend to generate repetitive and dull sequences, a phenomenon that is especially apparent when generating using greedy decoding. This issue persists even with state-of-the-art LLMs containing billions of parameters, trained via next-token prediction on large datasets. We find that by further fine-tuning these models to achieve a near-zero training loss on a small set of samples -- a process we refer to as hyperfitting -- the long-sequence generative capabilities are greatly enhanced. Greedy decoding with these Hyperfitted models even outperform Top-P sampling over long-sequences, both in terms of diversity and human preferences. This phenomenon extends to LLMs of various sizes, different domains, and even autoregressive image generation. We further find this phenomena to be distinctly different from that of Grokking and double descent. Surprisingly, our experiments indicate that hyperfitted models rarely fall into repeating sequences they were trained on, and even explicitly blocking these sequences results in high-quality output. All hyperfitted models produce extremely low-entropy predictions, often allocating nearly all probability to a single token.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19320v1">Shh, don't say that! Domain Certification in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 10 pages, includes appendix Published in International Conference on Learning Representations (ICLR) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often deployed to perform constrained tasks, with narrow domains. For example, customer support bots can be built on top of LLMs, relying on their broad language understanding and capabilities to enhance performance. However, these LLMs are adversarially susceptible, potentially generating outputs outside the intended domain. To formalize, assess, and mitigate this risk, we introduce domain certification; a guarantee that accurately characterizes the out-of-domain behavior of language models. We then propose a simple yet effective approach, which we call VALID that provides adversarial bounds as a certificate. Finally, we evaluate our method across a diverse set of datasets, demonstrating that it yields meaningful certificates, which bound the probability of out-of-domain samples tightly with minimum penalty to refusal behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19312v1">FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Website: https://fewshot-preference-optimization.github.io/
    </div>
    <details class="paper-abstract">
      Effective personalization of LLMs is critical for a broad range of user-interfacing applications such as virtual assistants and content curation. Inspired by the strong in-context learning capabilities of LLMs, we propose Few-Shot Preference Optimization (FSPO), which reframes reward modeling as a meta-learning problem. Under this framework, an LLM learns to quickly adapt to a user via a few labeled preferences from that user, constructing a personalized reward function for them. Additionally, since real-world preference data is scarce and challenging to collect at scale, we propose careful design choices to construct synthetic preference datasets for personalization, generating over 1M synthetic personalized preferences using publicly available LLMs. In particular, to successfully transfer from synthetic data to real users, we find it crucial for the data to exhibit both high diversity and coherent, self-consistent structure. We evaluate FSPO on personalized open-ended generation for up to 1,500 synthetic users across across three domains: movie reviews, pedagogical adaptation based on educational background, and general question answering, along with a controlled human study. Overall, FSPO achieves an 87% Alpaca Eval winrate on average in generating responses that are personalized to synthetic users and a 72% winrate with real human users in open-ended question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19301v1">Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) should undergo rigorous audits to identify potential risks, such as copyright and privacy infringements. Once these risks emerge, timely updates are crucial to remove undesirable responses, ensuring legal and safe model usage. It has spurred recent research into LLM unlearning, focusing on erasing targeted undesirable knowledge without compromising the integrity of other, non-targeted responses. Existing studies have introduced various unlearning objectives to pursue LLM unlearning without necessitating complete retraining. However, each of these objectives has unique properties, and no unified framework is currently available to comprehend them thoroughly. To fill the gap, we propose a toolkit of the gradient effect (G-effect), quantifying the impacts of unlearning objectives on model performance from a gradient perspective. A notable advantage is its broad ability to detail the unlearning impacts from various aspects across instances, updating steps, and LLM layers. Accordingly, the G-effect offers new insights into identifying drawbacks of existing unlearning objectives, further motivating us to explore a series of new solutions for their mitigation and improvements. Finally, we outline promising directions that merit further studies, aiming at contributing to the community to advance this important field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19295v1">Complex LLM Planning via Automated Heuristics Discovery</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      We consider enhancing large language models (LLMs) for complex planning tasks. While existing methods allow LLMs to explore intermediate steps to make plans, they either depend on unreliable self-verification or external verifiers to evaluate these steps, which demand significant data and computations. Here, we propose automated heuristics discovery (AutoHD), a novel approach that enables LLMs to explicitly generate heuristic functions to guide inference-time search, allowing accurate evaluation of intermediate states. These heuristic functions are further refined through a heuristic evolution process, improving their robustness and effectiveness. Our proposed method requires no additional model training or fine-tuning, and the explicit definition of heuristic functions generated by the LLMs provides interpretability and insights into the reasoning process. Extensive experiments across diverse benchmarks demonstrate significant gains over multiple baselines, including nearly twice the accuracy on some datasets, establishing our approach as a reliable and interpretable solution for complex planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02764v2">Drawing Pandas: A Benchmark for LLMs in Generating Plotting Code</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      This paper introduces the human-curated PandasPlotBench dataset, designed to evaluate language models' effectiveness as assistants in visual data exploration. Our benchmark focuses on generating code for visualizing tabular data - such as a Pandas DataFrame - based on natural language instructions, complementing current evaluation tools and expanding their scope. The dataset includes 175 unique tasks. Our experiments assess several leading Large Language Models (LLMs) across three visualization libraries: Matplotlib, Seaborn, and Plotly. We show that the shortening of tasks has a minimal effect on plotting capabilities, allowing for the user interface that accommodates concise user input without sacrificing functionality or accuracy. Another of our findings reveals that while LLMs perform well with popular libraries like Matplotlib and Seaborn, challenges persist with Plotly, highlighting areas for improvement. We hope that the modular design of our benchmark will broaden the current studies on generating visualizations. Our dataset and benchmark code are available online: https://huggingface.co/datasets/JetBrains-Research/PandasPlotBench; https://github.com/JetBrains-Research/PandasPlotBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19211v1">Negation-Induced Forgetting in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 ISCA/ITG Workshop on Diversity in Large Speech and Language Models
    </div>
    <details class="paper-abstract">
      The study explores whether Large Language Models (LLMs) exhibit negation-induced forgetting (NIF), a cognitive phenomenon observed in humans where negating incorrect attributes of an object or event leads to diminished recall of this object or event compared to affirming correct attributes (Mayo et al., 2014; Zang et al., 2023). We adapted Zang et al. (2023) experimental framework to test this effect in ChatGPT-3.5, GPT-4o mini and Llama3-70b-instruct. Our results show that ChatGPT-3.5 exhibits NIF, with negated information being less likely to be recalled than affirmed information. GPT-4o-mini showed a marginally significant NIF effect, while LLaMA-3-70B did not exhibit NIF. The findings provide initial evidence of negation-induced forgetting in some LLMs, suggesting that similar cognitive biases may emerge in these models. This work is a preliminary step in understanding how memory-related phenomena manifest in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19178v1">UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 10 pages, 3 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve remarkable success in natural language processing (NLP). In practical scenarios like recommendations, as users increasingly seek personalized experiences, it becomes crucial to incorporate user interaction history into the context of LLMs to enhance personalization. However, from a practical utility perspective, user interactions' extensive length and noise present challenges when used directly as text prompts. A promising solution is to compress and distill interactions into compact embeddings, serving as soft prompts to assist LLMs in generating personalized responses. Although this approach brings efficiency, a critical concern emerges: Can user embeddings adequately capture valuable information and prompt LLMs? To address this concern, we propose \name, a benchmark designed to evaluate the effectiveness of user embeddings in prompting LLMs for personalization. We establish a fair and standardized evaluation process, encompassing pre-training, fine-tuning, and evaluation stages. To thoroughly evaluate user embeddings, we design three dimensions of tasks: sequence understanding, action prediction, and interest perception. These evaluation tasks cover the industry's demands in traditional recommendation tasks, such as improving prediction accuracy, and its aspirations for LLM-based methods, such as accurately understanding user interests and enhancing the user experience. We conduct extensive experiments on various state-of-the-art methods for modeling user embeddings. Additionally, we reveal the scaling laws of leveraging user embeddings to prompt LLMs. The benchmark is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19159v1">A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Compared to width-wise pruning, depth-wise pruning can significantly accelerate inference in resource-constrained scenarios. Howerver, treating the entire Transformer layer as the minimum pruning unit may degrade model performance by indiscriminately discarding the entire information of the layer. This paper reveals the "Patch-like" feature relationship between layers in large language models by analyzing the correlation of the outputs of different layers in the reproducing kernel Hilbert space. Building on this observation, we proposes a sliding layer merging method that dynamically selects and fuses consecutive layers from top to bottom according to a pre-defined similarity threshold, thereby simplifying the model structure while maintaining its performance. Extensive experiments on LLMs with various architectures and different parameter scales show that our method outperforms existing pruning techniques in both zero-shot inference performance and retraining recovery quality after pruning. In particular, in the experiment with 35\% pruning on the Vicuna-7B model, our method achieved a 1.654\% improvement in average performance on zero-shot tasks compared to the existing method. Moreover, we further reveal the potential of combining depth pruning with width pruning to enhance the pruning effect. Our codes are available at https://github.com/920927/SLM-a-sliding-layer-merging-method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19149v1">Isolating Language-Coding from Problem-Solving: Benchmarking LLMs with PseudoEval</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Existing code generation benchmarks for Large Language Models (LLMs) such as HumanEval and MBPP are designed to study LLMs' end-to-end performance, where the benchmarks feed a problem description in natural language as input and examine the generated code in specific programming languages. However, the evaluation scores revealed in this way provide a little hint as to the bottleneck of the code generation -- whether LLMs are struggling with their problem-solving capability or language-coding capability. To answer this question, we construct PseudoEval, a multilingual code generation benchmark that provides a solution written in pseudocode as input. By doing so, the bottleneck of code generation in various programming languages could be isolated and identified. Our study yields several interesting findings. For example, we identify that the bottleneck of LLMs in Python programming is problem-solving, while Rust is struggling relatively more in language-coding. Also, our study indicates that problem-solving capability may transfer across programming languages, while language-coding needs more language-specific effort, especially for undertrained programming languages. Finally, we release the pipeline of constructing PseudoEval to facilitate the extension to existing benchmarks. PseudoEval is available at: https://anonymous.4open.science/r/PseudocodeACL25-7B74.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19148v1">Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Accepted by ICLR 2025, Project page: https://zowiezhang.github.io/projects/Amulet
    </div>
    <details class="paper-abstract">
      How to align large language models (LLMs) with user preferences from a static general dataset has been frequently studied. However, user preferences are usually personalized, changing, and diverse regarding culture, values, or time. This leads to the problem that the actual user preferences often do not coincide with those trained by the model developers in the practical use of LLMs. Since we cannot collect enough data and retrain for every demand, researching efficient real-time preference adaptation methods based on the backbone LLMs during test time is important. To this end, we introduce Amulet, a novel, training-free framework that formulates the decoding process of every token as a separate online learning problem with the guidance of simple user-provided prompts, thus enabling real-time optimization to satisfy users' personalized preferences. To reduce the computational cost brought by this optimization process for each token, we additionally provide a closed-form solution for each iteration step of the optimization process, thereby reducing the computational time cost to a negligible level. The detailed experimental results demonstrate that Amulet can achieve significant performance improvements in rich settings with combinations of different LLMs, datasets, and user preferences, while maintaining acceptable computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00034v2">Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for Diverse Hallucinations Categories</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 ACM TheWebConf 2025 Conference (WWW 2025) Research Track
    </div>
    <details class="paper-abstract">
      Recent studies have indicated that Large Language Models (LLMs) harbor an inherent understanding of truthfulness, yet often fail to consistently express it and generate false statements. This gap between "knowing" and "telling" poses a challenge for ensuring the truthfulness of generated content. Inspired by recent work on the practice of encoding human-interpretable concepts linearly within large language models, we treat truthfulness as a specially linearly encoded concept within LLMs, and introduce Adaptive Activation Steering (ACT), a tuning-free method that adaptively shifts LLM's activations in the "truthful" direction during inference. ACT addresses diverse categories of hallucinations by utilizing diverse truthfulness-related steering vectors and adjusting the steering intensity adaptively. Applied as an add-on across various models, ACT significantly improves truthfulness in LLaMA ($\uparrow$ 142%), LLaMA2 ($\uparrow$ 24%), Alpaca ($\uparrow$ 36%), Vicuna ($\uparrow$ 28%), LLaMA2-Chat ($\uparrow$ 19%), and LLaMA3($\uparrow$ 34%). Furthermore, we verify ACT's scalability across larger models (13B, 33B, 65B), underscoring the adaptability of ACT to large-scale language models. Our code is available at https://github.com/tianlwang/ACT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19135v1">A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Management</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      This paper presents a novel framework, called PLANTOR (PLanning with Natural language for Task-Oriented Robots), that integrates Large Language Models (LLMs) with Prolog-based knowledge management and planning for multi-robot tasks. The system employs a two-phase generation of a robot-oriented knowledge base, ensuring reusability and compositional reasoning, as well as a three-step planning procedure that handles temporal dependencies, resource constraints, and parallel task execution via mixed-integer linear programming. The final plan is converted into a Behaviour Tree for direct use in ROS2. We tested the framework in multi-robot assembly tasks within a block world and an arch-building scenario. Results demonstrate that LLMs can produce accurate knowledge bases with modest human feedback, while Prolog guarantees formal correctness and explainability. This approach underscores the potential of LLM integration for advanced robotics tasks requiring flexible, scalable, and human-understandable planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19133v1">DBox: Scaffolding Algorithmic Programming Learning through Learner-LLM Co-Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Decomposition is a fundamental skill in algorithmic programming, requiring learners to break down complex problems into smaller, manageable parts. However, current self-study methods, such as browsing reference solutions or using LLM assistants, often provide excessive or generic assistance that misaligns with learners' decomposition strategies, hindering independent problem-solving and critical thinking. To address this, we introduce Decomposition Box (DBox), an interactive LLM-based system that scaffolds and adapts to learners' personalized construction of a step tree through a "learner-LLM co-decomposition" approach, providing tailored support at an appropriate level. A within-subjects study (N=24) found that compared to the baseline, DBox significantly improved learning gains, cognitive engagement, and critical thinking. Learners also reported a stronger sense of achievement and found the assistance appropriate and helpful for learning. Additionally, we examined DBox's impact on cognitive load, identified usage patterns, and analyzed learners' strategies for managing system errors. We conclude with design implications for future AI-powered tools to better support algorithmic programming education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01833v3">Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Accepted at USENIX Security 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19098v1">Language-Driven Opinion Dynamics in Agent-Based Simulations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 34 pages, journal submission
    </div>
    <details class="paper-abstract">
      Understanding how opinions evolve is crucial for addressing issues such as polarization, radicalization, and consensus in social systems. While much research has focused on identifying factors influencing opinion change, the role of language and argumentative fallacies remains underexplored. This paper aims to fill this gap by investigating how language - along with social dynamics - influences opinion evolution through LODAS, a Language-Driven Opinion Dynamics Model for Agent-Based Simulations. The model simulates debates around the "Ship of Theseus" paradox, in which agents with discrete opinions interact with each other and evolve their opinions by accepting, rejecting, or ignoring the arguments presented. We study three different scenarios: balanced, polarized, and unbalanced opinion distributions. Agreeableness and sycophancy emerge as two main characteristics of LLM agents, and consensus around the presented statement emerges almost in any setting. Moreover, such AI agents are often producers of fallacious arguments in the attempt of persuading their peers and - for their complacency - they are also highly influenced by arguments built on logical fallacies. These results highlight the potential of this framework not only for simulating social dynamics but also for exploring from another perspective biases and shortcomings of LLMs, which may impact their interactions with humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19078v1">Sparse Brains are Also Adaptive Brains: Cognitive-Load-Aware Dynamic Activation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Dense large language models(LLMs) face critical efficiency bottlenecks as they rigidly activate all parameters regardless of input complexity. While existing sparsity methods(static pruning or dynamic activation) address this partially, they either lack adaptivity to contextual or model structural demands or incur prohibitive computational overhead. Inspired by human brain's dual-process mechanisms - predictive coding (N400) for backbone sparsity and structural reanalysis (P600) for complex context - we propose CLADA, a \textit{\textbf{C}ognitive-\textbf{L}oad-\textbf{A}ware \textbf{D}ynamic \textbf{A}ctivation} framework that synergizes statistical sparsity with semantic adaptability. Our key insight is that LLM activations exhibit two complementary patterns: 1) \textit{Global statistical sparsity} driven by sequence-level prefix information, and 2) \textit{Local semantic adaptability} modulated by cognitive load metrics(e.g., surprisal and entropy). CLADA employs a hierarchical thresholding strategy: a baseline from offline error-controlled optimization ensures 40\%+ sparsity, dynamically adjusted by real-time cognitive signals. Evaluations across six mainstream LLMs and nine benchmarks demonstrate that CLADA achieves \textbf{~20\% average speedup with <2\% accuracy drop}, outperforming Griffin (5\%+ degradation) and TT (negligible speedup). Crucially, we establish the first formal connection between neurolinguistic event-related potential (ERP) components and LLM efficiency mechanisms through multi-level regression analysis ($R^2=0.17$ for sparsity-adaptation synergy). Requiring no retraining or architectural changes, CLADA offers a deployable solution for resource-aware LLM inference while advancing biologically-inspired AI design. Our code is available at \href{https://github.com/Oldify/CLADA}{CLADA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19041v1">Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14182v2">LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 71 pages
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is revolutionizing scientific research, yet its growing integration into laboratory environments presents critical safety challenges. While large language models (LLMs) increasingly assist in tasks ranging from procedural guidance to autonomous experiment orchestration, an "illusion of understanding" may lead researchers to overestimate their reliability. Such overreliance is especially hazardous in high-stakes laboratory settings, where failures in hazard identification or risk assessment can result in severe accidents. To address these concerns, we propose the Laboratory Safety Benchmark (LabSafety Bench), a comprehensive framework that evaluates LLMs and vision language models (VLMs) on their ability to identify potential hazards, assess risks, and predict the consequences of unsafe actions in lab environments. LabSafety Bench comprises 765 multiple-choice questions aligned with US Occupational Safety and Health Administration (OSHA) protocols, along with 520 realistic laboratory scenarios featuring dual evaluation tasks: the Hazards Identification Test and the Consequence Identification Test, with 4090 open-ended questions in total. Evaluations across eight proprietary models, seven open-weight LLMs, and four VLMs reveal that, despite advanced performance on structured assessments, no model achieves the safety threshold required for reliable operation. None scored above 75% on the Hazards Identification Test. Moreover, while proprietary models tend to excel in multiple-choice evaluations, their performance in open-ended, real-world scenario responses is comparable to that of open-source models. These findings underscore the urgent need for specialized evaluation frameworks to ensure the safe and responsible deployment of AI in laboratory settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v3">MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 32 pages, 11 figures. v3 slightly adjust the author institution
    </div>
    <details class="paper-abstract">
      Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18910v1">CLLoRA: An Approach to Measure the Effects of the Context Length for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Large language model fine-tuning has been identified as an efficient approach to applying the pre-trained Large language models to other domains. To guarantee data privacy for different data owners, models are often fine-tuned in federated learning environments across different data owners, which often involve data heterogeneity issues and affect the fine-tuning performance. In addition, the length of the context for the training data has been identified as a major factor that affects the LLM's model performance. To efficiently measure how the context length affects the LLM's model performance in heterogeneous federated learning environments, we propose CLLoRA. CLLoRA utilizes the parameter-efficient fine-tuning approach LoRA based on different kinds of LLMs with varying sizes as the fine-tuning approach to investigate whether the quality and length of contexts can serve as standards for measuring non-IID context. The findings indicate that an imbalance in context quality not only affects local training on clients but also impacts the global model's performance. However, context length has a minimal effect on local training but a more significant influence on the global model. These results provide insights into how context quality and length affect the model performance for LLM fine-tuning in federated learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18904v1">An Empirical Study on Commit Message Generation using LLMs via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Accepted by the 47th IEEE/ACM International Conference on Software Engineering (ICSE'25)
    </div>
    <details class="paper-abstract">
      Commit messages concisely describe code changes in natural language and are important for software maintenance. Several approaches have been proposed to automatically generate commit messages, but they still suffer from critical limitations, such as time-consuming training and poor generalization ability. To tackle these limitations, we propose to borrow the weapon of large language models (LLMs) and in-context learning (ICL). Our intuition is based on the fact that the training corpora of LLMs contain extensive code changes and their pairwise commit messages, which makes LLMs capture the knowledge about commits, while ICL can exploit the knowledge hidden in the LLMs and enable them to perform downstream tasks without model tuning. However, it remains unclear how well LLMs perform on commit message generation via ICL. In this paper, we conduct an empirical study to investigate the capability of LLMs to generate commit messages via ICL. Specifically, we first explore the impact of different settings on the performance of ICL-based commit message generation. We then compare ICL-based commit message generation with state-of-the-art approaches on a popular multilingual dataset and a new dataset we created to mitigate potential data leakage. The results show that ICL-based commit message generation significantly outperforms state-of-the-art approaches on subjective evaluation and achieves better generalization ability. We further analyze the root causes for LLM's underperformance and propose several implications, which shed light on future research directions for using LLMs to generate commit messages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18886v1">On Pruning State-Space LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Recent work proposed state-space models (SSMs) as an efficient alternative to transformer-based LLMs. Can these models be pruned to further reduce their computation costs? We adapt several pruning methods to the SSM structure, and apply them to four SSM-based LLMs across multiple tasks. We find that such models are quite robust to some pruning methods (e.g. WANDA), while using other methods lead to fast performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18881v1">Letters from Future Self: Augmenting the Letter-Exchange Exercise with LLM-based Agents to Enhance Young Adults' Career Exploration</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 21 pages, 9 figures, Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Young adults often encounter challenges in career exploration. Self-guided interventions, such as the letter-exchange exercise, where participants envision and adopt the perspective of their future selves by exchanging letters with their envisioned future selves, can support career development. However, the broader adoption of such interventions may be limited without structured guidance. To address this, we integrated Large Language Model (LLM)-based agents that simulate participants' future selves into the letter-exchange exercise and evaluated their effectiveness. A one-week experiment (N=36) compared three conditions: (1) participants manually writing replies to themselves from the perspective of their future selves (baseline), (2) future-self agents generating letters to participants, and (3) future-self agents engaging in chat conversations with participants. Results indicated that exchanging letters with future-self agents enhanced participants' engagement during the exercise, while overall benefits of the intervention on future orientation, career self-concept, and psychological support remained comparable across conditions. We discuss design implications for AI-augmented interventions for supporting young adults' career exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v2">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 8 figures
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18387v2">How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 31 pages, 9 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Search plays a fundamental role in problem-solving across various domains, with most real-world decision-making problems being solvable through systematic search. Drawing inspiration from recent discussions on search and learning, we systematically explore the complementary relationship between search and Large Language Models (LLMs) from three perspectives. First, we analyze how learning can enhance search efficiency and propose Search via Learning (SeaL), a framework that leverages LLMs for effective and efficient search. Second, we further extend SeaL to SeaL-C to ensure rigorous completeness during search. Our evaluation across three real-world planning tasks demonstrates that SeaL achieves near-perfect accuracy while reducing search spaces by up to 99.1% compared to traditional approaches. Finally, we explore how far LLMs are from real search by investigating whether they can develop search capabilities independently. Our analysis reveals that while current LLMs struggle with efficient search in complex problems, incorporating systematic search strategies significantly enhances their problem-solving capabilities. These findings not only validate the effectiveness of our approach but also highlight the need for improving LLMs' search abilities for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09008v3">SuperCorrect: Advancing Small LLM Reasoning with Thought Template Distillation and Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 ICLR 2025. Project: https://github.com/YangLing0818/SuperCorrect-llm
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like GPT-4, DeepSeek-R1, and ReasonFlux have shown significant improvements in various reasoning tasks. However, smaller LLMs still struggle with complex mathematical reasoning because they fail to effectively identify and correct reasoning errors. Recent reflection-based methods aim to address these issues by enabling self-reflection and self-correction, but they still face challenges in independently detecting errors in their reasoning steps. To overcome these limitations, we propose SuperCorrect, a novel two-stage framework that uses a large teacher model to supervise and correct both the reasoning and reflection processes of a smaller student model. In the first stage, we extract hierarchical high-level and detailed thought templates from the teacher model to guide the student model in eliciting more fine-grained reasoning thoughts. In the second stage, we introduce cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training. This cross-model DPO approach teaches the student model to effectively locate and resolve erroneous thoughts with error-driven insights from the teacher model, breaking the bottleneck of its thoughts and acquiring new skills and knowledge to tackle challenging problems. Extensive experiments consistently demonstrate our superiority over previous methods. Notably, our SuperCorrect-7B model significantly surpasses powerful DeepSeekMath-7B by 7.8%/5.3% and Qwen2.5-Math-7B by 15.1%/6.3% on MATH/GSM8K benchmarks, achieving new SOTA performance among all 7B models. Code: https://github.com/YangLing0818/SuperCorrect-llm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18873v1">Multi-LLM Collaborative Search for Complex Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with complex reasoning tasks due to their limitations in addressing the vast reasoning space and inherent ambiguities of natural language. We propose the Mixture-of-Search-Agents (MoSA) paradigm, a novel approach leveraging the collective expertise of multiple LLMs to enhance search-based reasoning. MoSA integrates diverse reasoning pathways by combining independent exploration with iterative refinement among LLMs, mitigating the limitations of single-model approaches. Using Monte Carlo Tree Search (MCTS) as a backbone, MoSA enables multiple agents to propose and aggregate reasoning steps, resulting in improved accuracy. Our comprehensive evaluation across four reasoning benchmarks demonstrates MoSA's consistent performance improvements over single-agent and other multi-agent baselines, particularly in complex mathematical and commonsense reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18863v1">Sherlock: Towards Multi-scene Video Abnormal Event Extraction and Localization via a Global-local Spatial-sensitive LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Prior studies on Video Anomaly Detection (VAD) mainly focus on detecting whether each video frame is abnormal or not in the video, which largely ignore the structured video semantic information (i.e., what, when, and where does the abnormal event happen). With this in mind, we propose a new chat-paradigm \textbf{M}ulti-scene Video Abnormal Event Extraction and Localization (M-VAE) task, aiming to extract the abnormal event quadruples (i.e., subject, event type, object, scene) and localize such event. Further, this paper believes that this new task faces two key challenges, i.e., global-local spatial modeling and global-local spatial balancing. To this end, this paper proposes a Global-local Spatial-sensitive Large Language Model (LLM) named Sherlock, i.e., acting like Sherlock Holmes to track down the criminal events, for this M-VAE task. Specifically, this model designs a Global-local Spatial-enhanced MoE (GSM) module and a Spatial Imbalance Regulator (SIR) to address the two challenges respectively. Extensive experiments on our M-VAE instruction dataset show the significant advantages of Sherlock over several advanced Video-LLMs. This justifies the importance of global-local spatial information for the M-VAE task and the effectiveness of Sherlock in capturing such information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18862v1">Investigating Generalization of One-shot LLM Steering Vectors</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 20 pages, 7 figures. Code is available at https://github.com/jacobdunefsky/one-shot-steering-repro
    </div>
    <details class="paper-abstract">
      Steering vectors have emerged as a promising approach for interpreting and controlling LLMs, but current methods typically require large contrastive datasets that are often impractical to construct and may capture spurious correlations. We propose directly optimizing steering vectors through gradient descent on a single training example, and systematically investigate how these vectors generalize. We consider several steering optimization techniques, including multiple novel ones, and find that the resulting vectors effectively mediate safety-relevant behaviors in multiple models. Indeed, in experiments on an alignment-faking model, we are able to optimize one-shot steering vectors that induce harmful behavior on benign examples and whose negations suppress harmful behavior on malign examples. And in experiments on refusal suppression, we demonstrate that one-shot optimized steering vectors can transfer across inputs, yielding a Harmbench attack success rate of 96.9%. Furthermore, to quantitatively assess steering effectiveness in instruction-tuned models, we develop a novel evaluation framework using sequence probabilities from the corresponding base model. With this framework, we analyze how steering vectors modulate an instruction-tuned LLM's ability to recover from outputting false information, and find that this ability derives from the base model. Overall, our findings suggest that optimizing steering vectors on a single example can mediate misaligned behavior in LLMs, and provide a path toward better understanding the relationship between LLM behavior and activation space structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05864v2">Permute-and-Flip: An optimally stable and watermarkable decoder for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys stability properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-stability tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson (2023)'s Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and its Gumbel watermarked counterpart) in terms of perplexity, while retaining the same stability (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18836v1">REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 14 pages, 4 figures, 9 tables
    </div>
    <details class="paper-abstract">
      This benchmark suite provides a comprehensive evaluation framework for assessing both individual LLMs and multi-agent systems in real-world planning scenarios. The suite encompasses eleven designed problems that progress from basic to highly complex, incorporating key aspects such as multi-agent coordination, inter-agent dependencies, and dynamic environmental disruptions. Each problem can be scaled along three dimensions: the number of parallel planning threads, the complexity of inter-dependencies, and the frequency of unexpected disruptions requiring real-time adaptation. The benchmark includes detailed specifications, evaluation metrics, and baseline implementations using contemporary frameworks like LangGraph, enabling rigorous testing of both single-agent and multi-agent planning capabilities. Through standardized evaluation criteria and scalable complexity, this benchmark aims to drive progress in developing more robust and adaptable AI planning systems for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18822v1">Data-Efficient Multi-Agent Spatial Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      In this project, our goal is to determine how to leverage the world-knowledge of pretrained large language models for efficient and robust learning in multiagent decision making. We examine this in a taxi routing and assignment problem where agents must decide how to best pick up passengers in order to minimize overall waiting time. While this problem is situated on a graphical road network, we show that with the proper prompting zero-shot performance is quite strong on this task. Furthermore, with limited fine-tuning along with the one-at-a-time rollout algorithm for look ahead, LLMs can out-compete existing approaches with 50 times fewer environmental interactions. We also explore the benefits of various linguistic prompting approaches and show that including certain easy-to-compute information in the prompt significantly improves performance. Finally, we highlight the LLM's built-in semantic understanding, showing its ability to adapt to environmental factors through simple prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18810v1">Holistic Audit Dataset Generation for LLM Unlearning via Knowledge Graph Traversal and Redundancy Removal</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have faced increasing demands to selectively remove sensitive information, protect privacy, and comply with copyright regulations through unlearning, by Machine Unlearning. While evaluating unlearning effectiveness is crucial, existing benchmarks are limited in scale and comprehensiveness, typically containing only a few hundred test cases. We identify two critical challenges in generating holistic audit datasets: ensuring audit adequacy and handling knowledge redundancy between forget and retain dataset. To address these challenges, we propose HANKER, an automated framework for holistic audit dataset generation leveraging knowledge graphs to achieve fine-grained coverage and eliminate redundant knowledge. Applying HANKER to the popular MUSE benchmark, we successfully generated over 69,000 and 111,000 audit cases for the News and Books datasets respectively, identifying thousands of knowledge memorization instances that the previous benchmark failed to detect. Our empirical analysis uncovers how knowledge redundancy significantly skews unlearning effectiveness metrics, with redundant instances artificially inflating the observed memorization measurements ROUGE from 19.7% to 26.1% and Entailment Scores from 32.4% to 35.2%, highlighting the necessity of systematic deduplication for accurate assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12022v2">Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18791v1">Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 21 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The surge of LLM studies makes synthesizing their findings challenging. Meta-analysis can uncover important trends across studies, but its use is limited by the time-consuming nature of manual data extraction. Our study presents a semi-automated approach for meta-analysis that accelerates data extraction using LLMs. It automatically identifies relevant arXiv papers, extracts experimental results and related attributes, and organizes them into a structured dataset. We conduct a comprehensive meta-analysis of frontier LLMs using an automatically extracted dataset, reducing the effort of paper surveying and data extraction by more than 93\% compared to manual approaches. We validate our dataset by showing that it reproduces key findings from a recent manual meta-analysis about Chain-of-Thought (CoT), and also uncovers new insights that go beyond it, showing for example that in-context examples benefit multimodal tasks but offer limited gains in mathematical tasks compared to CoT. Our automatically updatable dataset enables continuous tracking of target models by extracting evaluation studies as new data becomes available. Through our scientific artifacts and empirical analysis, we provide novel insights into LLMs while facilitating ongoing meta-analyses of their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08197v2">From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 ICLR 2025 Oral;GitHub:https://github.com/quchangle1/DRAFT
    </div>
    <details class="paper-abstract">
      Tool learning enables Large Language Models (LLMs) to interact with external environments by invoking tools, serving as an effective strategy to mitigate the limitations inherent in their pre-training data. In this process, tool documentation plays a crucial role by providing usage instructions for LLMs, thereby facilitating effective tool utilization. This paper concentrates on the critical challenge of bridging the comprehension gap between LLMs and external tools due to the inadequacies and inaccuracies inherent in existing human-centric tool documentation. We propose a novel framework, DRAFT, aimed at Dynamically Refining tool documentation through the Analysis of Feedback and Trials emanating from LLMs' interactions with external tools. This methodology pivots on an innovative trial-and-error approach, consisting of three distinct learning phases: experience gathering, learning from experience, and documentation rewriting, to iteratively enhance the tool documentation. This process is further optimized by implementing a diversity-promoting exploration strategy to ensure explorative diversity and a tool-adaptive termination mechanism to prevent overfitting while enhancing efficiency. Extensive experiments on multiple datasets demonstrate that DRAFT's iterative, feedback-based refinement significantly ameliorates documentation quality, fostering a deeper comprehension and more effective utilization of tools by LLMs. Notably, our analysis reveals that the tool documentation refined via our approach demonstrates robust cross-model generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19381v5">HYBRIDMIND: Meta Selection of Natural Language and Symbolic Language for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      LLMs approach logical and mathematical reasoning through natural or symbolic languages. While natural language offers human-accessible flexibility but suffers from ambiguity, symbolic reasoning provides precise, machine-executable inferences at the cost of strict domain constraints. We introduce HYBRIDMIND, an adaptive strategy that selects the optimal reasoning approach for each reasoning problem. Through extensive experiments, we evaluate both prompting-based approaches with state-of-the-art LLMs and fine-tuned open-source models. We find that fine-tuning LLaMA-3.1-8B-Instruct as a meta-selector outperforms GPT-4o's natural language reasoning by 4.4\% on FOLIO and 1.3\% on MATH. More notably, using GPT-3.5-turbo as a prompted meta-selector yields a 10\% improvement on FOLIO's challenging subset compared to GPT-4o. We will release our code and data to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18771v1">Exploring Graph Tasks with Pure LLMs: A Comprehensive Benchmark and Investigation</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Graph-structured data has become increasingly prevalent across various domains, raising the demand for effective models to handle graph tasks like node classification and link prediction. Traditional graph learning models like Graph Neural Networks (GNNs) have made significant strides, but their capabilities in handling graph data remain limited in certain contexts. In recent years, large language models (LLMs) have emerged as promising candidates for graph tasks, yet most studies focus primarily on performance benchmarks and fail to address their broader potential, including their ability to handle limited data, their transferability across tasks, and their robustness. In this work, we provide a comprehensive exploration of LLMs applied to graph tasks. We evaluate the performance of pure LLMs, including those without parameter optimization and those fine-tuned with instructions, across various scenarios. Our analysis goes beyond accuracy, assessing LLM ability to perform in few-shot/zero-shot settings, transfer across domains, understand graph structures, and demonstrate robustness in challenging scenarios. We conduct extensive experiments with 16 graph learning models alongside 6 LLMs (e.g., Llama3B, GPT-4o, Qwen-plus), comparing their performance on datasets like Cora, PubMed, ArXiv, and Products. Our findings show that LLMs, particularly those with instruction tuning, outperform traditional models in few-shot settings, exhibit strong domain transferability, and demonstrate excellent generalization and robustness. This work offers valuable insights into the capabilities of LLMs for graph learning, highlighting their advantages and potential for real-world applications, and paving the way for future research in this area. Codes and datasets are released in https://github.com/myflashbarry/LLM-benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05127v4">Towards Semantic Equivalence of Tokenization in Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 ICLR-2025. The project page: https://chocowu.github.io/SeTok-web/
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in processing vision-language tasks. One of the crux of MLLMs lies in vision tokenization, which involves efficiently transforming input visual signals into feature representations that are most beneficial for LLMs. However, existing vision tokenizers, essential for semantic alignment between vision and language, remain problematic. Existing methods aggressively fragment visual input, corrupting the visual semantic integrity. To address this, this paper proposes a novel dynamic Semantic-Equivalent Vision Tokenizer (SeTok), which groups visual features into semantic units via a dynamic clustering algorithm, flexibly determining the number of tokens based on image complexity. The resulting vision tokens effectively preserve semantic integrity and capture both low-frequency and high-frequency visual features. The proposed MLLM (Setokim) equipped with SeTok significantly demonstrates superior performance across various tasks, as evidenced by our experimental results. The project page is at https://chocowu.github.io/SeTok-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04077v2">AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      With the development of large language models (LLMs), efficient inference through Key-Value (KV) cache compression has attracted considerable attention, especially for long-context generation. To compress the KV cache, recent methods identify critical KV tokens through heuristic ranking with attention scores. However, these methods often struggle to accurately determine critical tokens as they neglect the \textit{temporal patterns} in attention scores, resulting in a noticeable degradation in LLM performance. To address this challenge, we propose AttentionPredictor, which is the first learning-based critical token identification approach. Specifically, AttentionPredictor learns a lightweight convolution model to capture spatiotemporal patterns and predict the next-token attention score. An appealing feature of AttentionPredictor is that it accurately predicts the attention score while consuming negligible memory. Moreover, we propose a cross-token critical cache prefetching framework that hides the token estimation time overhead to accelerate the decoding stage. By retaining most of the attention information, AttentionPredictor achieves 16$\times$ KV cache compression with comparable LLM performance, significantly outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14804v4">Can LLMs Solve longer Math Word Problems Better?</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18755v1">M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are one of the most important killer computer applications. The recent algorithmic advancement proposes a fine-grained group-wise quantization for LLMs, which treats a small set (e.g., 64) of values in a tensor as a compression unit. It effectively preserves the model accuracy without retraining, and has become the standard approach to efficiently deploy LLMs. On the other hand, there are works that propose various adaptive data types to better adapt to different distributions and further reduce the required bit length for LLMs. In this work, our detailed analysis unveils a key finding that while different tensors exhibit similar distributions, small groups can have markedly different distributions. As such, the group-level diversity requires a new level of adaptivity for which existing adaptive data types fail to provide. In this paper, we propose MANT, a mathematically adaptive numeric type, featuring a more flexible encoding paradigm with a wider range of data distribution and more efficient decodingcomputation fusion mechanism to address these challenges. Based on MANT, we develop a supporting framework to assign the appropriate data type for each group adaptively. Meanwhile, the dynamically generated Key-Value (KV) caches in LLMs introduce further complexity for real-time quantization. To tackle this, we propose an efficient real-time quantization mechanism. Besides, we implement a specific processing element (PE) to efficiently support MANT and incorporate a real-time quantization unit. By integrating these components into a systolic array, MANT unifies the group-wise weight and KV cache quantization and addresses the associated challenges. Our evaluation shows achieving, on average, 2.99x (up to 4.46x) speedup and 2.81x (up to 4.10x) energy reduction to the state-of-the-art LLM accelerator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18754v1">AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 8 pages, 10 figures, in Proceedings of the ACM Web Conference 2025 (WWW '25)
    </div>
    <details class="paper-abstract">
      The AgentSociety Challenge is the first competition in the Web Conference that aims to explore the potential of Large Language Model (LLM) agents in modeling user behavior and enhancing recommender systems on web platforms. The Challenge consists of two tracks: the User Modeling Track and the Recommendation Track. Participants are tasked to utilize a combined dataset from Yelp, Amazon, and Goodreads, along with an interactive environment simulator, to develop innovative LLM agents. The Challenge has attracted 295 teams across the globe and received over 1,400 submissions in total over the course of 37 official competition days. The participants have achieved 21.9% and 20.3% performance improvement for Track 1 and Track 2 in the Development Phase, and 9.1% and 15.9% in the Final Phase, representing a significant accomplishment. This paper discusses the detailed designs of the Challenge, analyzes the outcomes, and highlights the most successful LLM agent designs. To support further research and development, we have open-sourced the benchmark environment at https://tsinghua-fib-lab.github.io/AgentSocietyChallenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03688v2">A Comparison of DeepSeek and Other LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 21 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Recently, DeepSeek has been the focus of attention in and beyond the AI community. An interesting problem is how DeepSeek compares to other large language models (LLMs). There are many tasks an LLM can do, and in this paper, we use the task of predicting an outcome using a short text for comparison. We consider two settings, an authorship classification setting and a citation classification setting. In the first one, the goal is to determine whether a short text is written by human or AI. In the second one, the goal is to classify a citation to one of four types using the textual content. For each experiment, we compare DeepSeek with $4$ popular LLMs: Claude, Gemini, GPT, and Llama. We find that, in terms of classification accuracy, DeepSeek outperforms Gemini, GPT, and Llama in most cases, but underperforms Claude. We also find that DeepSeek is comparably slower than others but with a low cost to use, while Claude is much more expensive than all the others. Finally, we find that in terms of similarity, the output of DeepSeek is most similar to those of Gemini and Claude (and among all $5$ LLMs, Claude and Gemini have the most similar outputs). In this paper, we also present a fully-labeled dataset collected by ourselves, and propose a recipe where we can use the LLMs and a recent data set, MADStat, to generate new data sets. The datasets in our paper can be used as benchmarks for future study on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17410v2">COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 23 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but this leads to substantial approximation errors, resulting in less effective optimization (e.g., in terms of per-token efficiency). In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS. Our code is available at https://github.com/lliu606/COSMOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18712v1">TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Accepted WWW2025 Demo Paper
    </div>
    <details class="paper-abstract">
      This work leverages Large Language Models (LLMs) to simulate human mobility, addressing challenges like high costs and privacy concerns in traditional models. Our hierarchical framework integrates persona generation, activity selection, and destination prediction, using real-world demographic and psychological data to create realistic movement patterns. Both physical models and language models are employed to explore and demonstrate different methodologies for human mobility simulation. By structuring data with summarization and weighted density metrics, the system ensures scalable memory management while retaining actionable insights. Preliminary results indicate that LLM-driven simulations align with observed real-world patterns, offering scalable, interpretable insights for social problems such as urban planning, traffic management, and public health. The framework's ability to dynamically generate personas and activities enables it to provide adaptable and realistic daily routines. This study demonstrates the transformative potential of LLMs in advancing mobility modeling for societal and urban applications. The source code and interactive demo for our framework are available at https://github.com/cju0/TrajLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19622v1">Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 12 pages, 1 figure, 3 tables, 4 prompt/data templates
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have raised interest in their formal reasoning capabilities, particularly in mathematics. While closed LLMs like GPT-4 perform well on mathematical benchmarks, e.g., GSM8K, it remains unclear whether small to medium-sized open LLMs can achieve similar performance, questioning their reliability. To close this gap, we propose a post-training approach leveraging a mixture of opinions (MoO) from weaker ancillary LLMs to enhance a (relatively) stronger LLM's reasoning. For that, each post-training sample is augmented with Chain-of-Thought (CoT) reasoning steps and answers from ancillary LLMs, enabling the main LLM to learn from diverse perspectives. We compare MoO with standard supervised fine-tuning (SFT), few-shot prompting, and the Mixture of Agents (MoA) method on mathematical reasoning benchmarks. Our results show that incorporating weaker LLMs' opinions improves mathematical reasoning by an average of 5%, highlighting the value of diverse perspectives in reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19614v1">Is Your Paper Being Reviewed by an LLM? A New Benchmark Dataset and Approach for Detecting AI Text in Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Peer review is a critical process for ensuring the integrity of published scientific research. Confidence in this process is predicated on the assumption that experts in the relevant domain give careful consideration to the merits of manuscripts which are submitted for publication. With the recent rapid advancements in large language models (LLMs), a new risk to the peer review process is that negligent reviewers will rely on LLMs to perform the often time consuming process of reviewing a paper. However, there is a lack of existing resources for benchmarking the detectability of AI text in the domain of peer review. To address this deficiency, we introduce a comprehensive dataset containing a total of 788,984 AI-written peer reviews paired with corresponding human reviews, covering 8 years of papers submitted to each of two leading AI research conferences (ICLR and NeurIPS). We use this new resource to evaluate the ability of 18 existing AI text detection algorithms to distinguish between peer reviews written by humans and different state-of-the-art LLMs. Motivated by the shortcomings of existing methods, we propose a new detection approach which surpasses existing methods in the identification of AI written peer reviews. Our work reveals the difficulty of identifying AI-generated text at the individual peer review level, highlighting the urgent need for new tools and methods to detect this unethical use of generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19607v1">Revisiting Word Embeddings in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable advancement in various NLP tasks. As such, a popular trend has emerged lately where NLP researchers extract word/sentence/document embeddings from these large decoder-only models and use them for various inference tasks with promising results. However, it is still unclear whether the performance improvement of LLM-induced embeddings is merely because of scale or whether underlying embeddings they produce significantly differ from classical encoding models like Word2Vec, GloVe, Sentence-BERT (SBERT) or Universal Sentence Encoder (USE). This is the central question we investigate in the paper by systematically comparing classical decontextualized and contextualized word embeddings with the same for LLM-induced embeddings. Our results show that LLMs cluster semantically related words more tightly and perform better on analogy tasks in decontextualized settings. However, in contextualized settings, classical models like SimCSE often outperform LLMs in sentence-level similarity assessment tasks, highlighting their continued relevance for fine-grained semantics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19582v1">Where Are We? Evaluating LLM Performance on African Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Africa's rich linguistic heritage remains underrepresented in NLP, largely due to historical policies that favor foreign languages and create significant data inequities. In this paper, we integrate theoretical insights on Africa's language landscape with an empirical evaluation using Sahara - a comprehensive benchmark curated from large-scale, publicly accessible datasets capturing the continent's linguistic diversity. By systematically assessing the performance of leading large language models (LLMs) on Sahara, we demonstrate how policy-induced data variations directly impact model effectiveness across African languages. Our findings reveal that while a few languages perform reasonably well, many Indigenous languages remain marginalized due to sparse data. Leveraging these insights, we offer actionable recommendations for policy reforms and inclusive data practices. Overall, our work underscores the urgent need for a dual approach - combining theoretical understanding with empirical evaluation - to foster linguistic diversity in AI for African communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19571v1">LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training via Efficient Zeroth-Order Adaptive SAM</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      We study robust parameter-efficient fine-tuning (PEFT) techniques designed to improve accuracy and generalization while operating within strict computational and memory hardware constraints, specifically focusing on large-language models (LLMs). Existing PEFT methods often lack robustness and fail to generalize effectively across diverse tasks, leading to suboptimal performance in real-world scenarios. To address this, we present a new highly computationally efficient framework called AdaZo-SAM, combining Adam and Sharpness-Aware Minimization (SAM) while requiring only a single-gradient computation in every iteration. This is achieved using a stochastic zeroth-order estimation to find SAM's ascent perturbation. We provide a convergence guarantee for AdaZo-SAM and show that it improves the generalization ability of state-of-the-art PEFT methods. Additionally, we design a low-rank gradient optimization method named LORENZA, which is a memory-efficient version of AdaZo-SAM. LORENZA utilizes a randomized SVD scheme to efficiently compute the subspace projection matrix and apply optimization steps onto the selected subspace. This technique enables full-parameter fine-tuning with adaptive low-rank gradient updates, achieving the same reduced memory consumption as gradient-low-rank-projection methods. We provide a convergence analysis of LORENZA and demonstrate its merits for pre-training and fine-tuning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09073v2">CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 full draft: 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various NLP tasks but struggle with code-mixed (or code-switched) language understanding. For example, prior work benchmarking the performance of multilingual LLMs on code-mixed translation tasks has demonstrated that current state-of-the-art multilingual LLMs are ineffective in dealing with code-mixed languages. However, the question of how to improve the capability of multilingual LLMs to handle code-mixed language has not received any attention to date. In this paper, we tackle this research gap by proposing CHAI, a novel general-purpose framework for improving the ability of multilingual LLMs to handle code-mixed languages. CHAI relies on three novel contributions made in this paper. First, we explore the ability of LLMs to provide accurate annotations for code-mixed translation tasks. Second, we leverage this ability of LLMs as annotators to generate preference data for code-mixed translation tasks at scale, which are then used within a reinforcement learning from AI feedback (RLAIF) procedure to improve LLMs' capability on code-mixed tasks. Third, we conduct a rigorous experimental evaluation across various real-world datasets and settings. Our analysis shows that CHAI-powered LLMs outperform state-of-the-art open-source LLMs by 25.66% (in terms of win rate adjudicated by human annotators) in code-mixed translation tasks. This work represents a first step towards developing more inclusive code-mixed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19518v1">Accessing LLMs for Front-end Software Architecture Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 4 pages, 1 figure, to appear in the International Workshop on Designing Software at ICSE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant promise in automating software development tasks, yet their capabilities with respect to software design tasks remains largely unclear. This study investigates the capabilities of an LLM in understanding, reproducing, and generating structures within the complex VIPER architecture, a design pattern for iOS applications. We leverage Bloom's taxonomy to develop a comprehensive evaluation framework to assess the LLM's performance across different cognitive domains such as remembering, understanding, applying, analyzing, evaluating, and creating. Experimental results, using ChatGPT 4 Turbo 2024-04-09, reveal that the LLM excelled in higher-order tasks like evaluating and creating, but faced challenges with lower-order tasks requiring precise retrieval of architectural details. These findings highlight both the potential of LLMs to reduce development costs and the barriers to their effective application in real-world software design scenarios. This study proposes a benchmark format for assessing LLM capabilities in software architecture, aiming to contribute toward more robust and accessible AI-driven development tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19463v1">Do LLMs exhibit demographic parity in responses to queries about Human Rights?</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      This research describes a novel approach to evaluating hedging behaviour in large language models (LLMs), specifically in the context of human rights as defined in the Universal Declaration of Human Rights (UDHR). Hedging and non-affirmation are behaviours that express ambiguity or a lack of clear endorsement on specific statements. These behaviours are undesirable in certain contexts, such as queries about whether different groups are entitled to specific human rights; since all people are entitled to human rights. Here, we present the first systematic attempt to measure these behaviours in the context of human rights, with a particular focus on between-group comparisons. To this end, we design a novel prompt set on human rights in the context of different national or social identities. We develop metrics to capture hedging and non-affirmation behaviours and then measure whether LLMs exhibit demographic parity when responding to the queries. We present results on three leading LLMs and find that all models exhibit some demographic disparities in how they attribute human rights between different identity groups. Futhermore, there is high correlation between different models in terms of how disparity is distributed amongst identities, with identities that have high disparity in one model also facing high disparity in both the other models. While baseline rates of hedging and non-affirmation differ, these disparities are consistent across queries that vary in ambiguity and they are robust across variations of the precise query wording. Our findings highlight the need for work to explicitly align LLMs to human rights principles, and to ensure that LLMs endorse the human rights of all groups equally.
    </details>
</div>
