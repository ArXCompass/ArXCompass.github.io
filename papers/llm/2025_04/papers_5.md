# llm - 2025_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11969v4">Does Refusal Training in LLMs Generalize to the Past Tense?</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted at ICLR 2025. Updates in v2 and v3: added GPT-4o, Claude 3.5 Sonnet, o1-mini, and o1-preview results. Code and jailbreak artifacts: https://github.com/tml-epfl/llm-past-tense
    </div>
    <details class="paper-abstract">
      Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13272v1">Using LLMs for Library Migration</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Library migration is the process of replacing a used software library with another library that provides similar functionality. Manual library migration is time-consuming and error prone, as it requires developers to understand the APIs of both libraries, map them, and perform the necessary code transformations. Due to its difficulty, most of the existing automated techniques and tooling stop at the API mapping stage or support a limited set of code transformations. On the other hand, Large Language Models (LLMs) are good at generating and transforming code and finding similar code, which are necessary upstream tasks for library migration. Such capabilities suggest that LLMs may be suitable for library migration. Therefore, in this paper, we investigate the effectiveness of LLMs for migration between Python libraries. We evaluate three LLMs, LLama 3.1, GPT-4o mini, and GPT-4o on PyMigBench, where we migrate 321 real-world library migrations that include 2,989 migration-related code changes. We measure the correctness of the migration results in two ways. We first compare the LLM's migrated code with the developers' migrated code in the benchmark and then run the unit tests available in the client repositories. We find that LLama 3.1, GPT-4o mini, and GPT-4o correctly migrate 89%, 89%, and 94% of the migration-related code changes. respectively. We also find that 36%, 52% and 64% of the LLama 3.1, GPT-4o mini, and GPT-4o migrations pass the same tests that passed in the developer's migration. Overall, our results suggest that LLMs can be effective in migrating code between libraries, but we also identify cases that pose difficulties for the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13217v1">Sustainability via LLM Right-sizing</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 17 pages, 2 Figures, 6 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly embedded in organizational workflows. This has raised concerns over their energy consumption, financial costs, and data sovereignty. While performance benchmarks often celebrate cutting-edge models, real-world deployment decisions require a broader perspective: when is a smaller, locally deployable model "good enough"? This study offers an empirical answer by evaluating eleven proprietary and open-weight LLMs across ten everyday occupational tasks, including summarizing texts, generating schedules, and drafting emails and proposals. Using a dual-LLM-based evaluation framework, we automated task execution and standardized evaluation across ten criteria related to output quality, factual accuracy, and ethical responsibility. Results show that GPT-4o delivers consistently superior performance but at a significantly higher cost and environmental footprint. Notably, smaller models like Gemma-3 and Phi-4 achieved strong and reliable results on most tasks, suggesting their viability in contexts requiring cost-efficiency, local deployment, or privacy. A cluster analysis revealed three model groups -- premium all-rounders, competent generalists, and limited but safe performers -- highlighting trade-offs between quality, control, and sustainability. Significantly, task type influenced model effectiveness: conceptual tasks challenged most models, while aggregation and transformation tasks yielded better performances. We argue for a shift from performance-maximizing benchmarks to task- and context-aware sufficiency assessments that better reflect organizational priorities. Our approach contributes a scalable method to evaluate AI models through a sustainability lens and offers actionable guidance for responsible LLM deployment in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15299v1">D$^{2}$MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted by MobiCom 2025
    </div>
    <details class="paper-abstract">
      The mixture of experts (MoE) model is a sparse variant of large language models (LLMs), designed to hold a better balance between intelligent capability and computational overhead. Despite its benefits, MoE is still too expensive to deploy on resource-constrained edge devices, especially with the demands of on-device inference services. Recent research efforts often apply model compression techniques, such as quantization, pruning and merging, to restrict MoE complexity. Unfortunately, due to their predefined static model optimization strategies, they cannot always achieve the desired quality-overhead trade-off when handling multiple requests, finally degrading the on-device quality of service. These limitations motivate us to propose the D$^2$MoE, an algorithm-system co-design framework that matches diverse task requirements by dynamically allocating the most proper bit-width to each expert. Specifically, inspired by the nested structure of matryoshka dolls, we propose the matryoshka weight quantization (MWQ) to progressively compress expert weights in a bit-nested manner and reduce the required runtime memory. On top of it, we further optimize the I/O-computation pipeline and design a heuristic scheduling algorithm following our hottest-expert-bit-first (HEBF) principle, which maximizes the expert parallelism between I/O and computation queue under constrained memory budgets, thus significantly reducing the idle temporal bubbles waiting for the experts to load. Evaluations on real edge devices show that D$^2$MoE improves the overall inference throughput by up to 1.39$\times$ and reduces the peak memory footprint by up to 53% over the latest on-device inference frameworks, while still preserving comparable serving accuracy as its INT8 counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13145v1">Exploring Expert Failures Improves LLM Agent Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown tremendous potential as agents, excelling at tasks that require multiple rounds of reasoning and interactions. Rejection Sampling Fine-Tuning (RFT) has emerged as an effective method for finetuning LLMs as agents: it first imitates expert-generated successful trajectories and further improves agentic skills through iterative fine-tuning on successful, self-generated trajectories. However, since the expert (e.g., GPT-4) succeeds primarily on simpler subtasks and RFT inherently favors simpler scenarios, many complex subtasks remain unsolved and persistently out-of-distribution (OOD). Upon investigating these challenging subtasks, we discovered that previously failed expert trajectories can often provide valuable guidance, e.g., plans and key actions, that can significantly improve agent exploration efficiency and acquisition of critical skills. Motivated by these observations, we propose Exploring Expert Failures (EEF), which identifies beneficial actions from failed expert trajectories and integrates them into the training dataset. Potentially harmful actions are meticulously excluded to prevent contamination of the model learning process. By leveraging the beneficial actions in expert failures, EEF successfully solves some previously unsolvable subtasks and improves agent tuning performance. Remarkably, our approach achieved a 62\% win rate in WebShop, outperforming RFT (53. 6\%) and GPT-4 (35. 6\%), and to the best of our knowledge, setting a new state-of-the-art as the first method to surpass a score of 0.81 in WebShop and exceed 81 in SciWorld.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13125v1">LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      This paper investigates the application of large language models (LLMs) to financial tasks. We fine-tuned foundation models using the Open FinLLM Leaderboard as a benchmark. Building on Qwen2.5 and Deepseek-R1, we employed techniques including supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) to enhance their financial capabilities. The fine-tuned models demonstrated substantial performance gains across a wide range of financial tasks. Moreover, we measured the data scaling law in the financial domain. Our work demonstrates the potential of large language models (LLMs) in financial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16063v4">Citation-Enhanced Generation for LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit powerful general intelligence across diverse scenarios, including their integration into chatbots. However, a vital challenge of LLM-based chatbots is that they may produce hallucinated content in responses, which significantly limits their applicability. Various efforts have been made to alleviate hallucination, such as retrieval augmented generation and reinforcement learning with human feedback, but most of them require additional training and data annotation. In this paper, we propose a novel post-hoc Citation-Enhanced Generation (CEG) approach combined with retrieval argumentation. Unlike previous studies that focus on preventing hallucinations during generation, our method addresses this issue in a post-hoc way. It incorporates a retrieval module to search for supporting documents relevant to the generated content, and employs a natural language inference-based citation generation module. Once the statements in the generated content lack of reference, our model can regenerate responses until all statements are supported by citations. Note that our method is a training-free plug-and-play plugin that is capable of various LLMs. Experiments on various hallucination-related datasets show our framework outperforms state-of-the-art methods in both hallucination detection and response regeneration on three benchmarks. Our codes and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11514v3">Towards Translating Real-World Code with LLMs: A Study of Translating to Rust</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 12 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in code translation - the task of translating code written in one programming language to another language - due to their ability to write code in most programming languages. However, LLM's effectiveness on translating real-world code remains largely unstudied. In this work, we perform the first substantial study on LLM-based translation to Rust by assessing the ability of five state-of-the-art LLMs, GPT4, Claude 3, Claude 2.1, Gemini Pro, and Mixtral. We conduct our study on code extracted from real-world open source projects. To enable our study, we develop FLOURINE, an end-to-end code translation tool that uses differential fuzzing to check if a Rust translation is I/O equivalent to the original source program, eliminating the need for pre-existing test cases. As part of our investigation, we assess both the LLM's ability to produce an initially successful translation, as well as their capacity to fix a previously generated buggy one. If the original and the translated programs are not I/O equivalent, we apply a set of automated feedback strategies, including feedback to the LLM with counterexamples. Our results show that the most successful LLM can translate 47% of our benchmarks, and also provides insights into next steps for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09012v2">Multimodal LLMs Can Reason about Aesthetics in Zero-Shot</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 WIP, Homepage https://github.com/songrise/MLLM4Art
    </div>
    <details class="paper-abstract">
      The rapid progress of generative art has democratized the creation of visually pleasing imagery. However, achieving genuine artistic impact - the kind that resonates with viewers on a deeper, more meaningful level - requires a sophisticated aesthetic sensibility. This sensibility involves a multi-faceted reasoning process extending beyond mere visual appeal, which is often overlooked by current computational models. This paper pioneers an approach to capture this complex process by investigating how the reasoning capabilities of Multimodal LLMs (MLLMs) can be effectively elicited for aesthetic judgment. Our analysis reveals a critical challenge: MLLMs exhibit a tendency towards hallucinations during aesthetic reasoning, characterized by subjective opinions and unsubstantiated artistic interpretations. We further demonstrate that these limitations can be overcome by employing an evidence-based, objective reasoning process, as substantiated by our proposed baseline, ArtCoT. MLLMs prompted by this principle produce multi-faceted and in-depth aesthetic reasoning that aligns significantly better with human judgment. These findings have direct applications in areas such as AI art tutoring and as reward models for generative art. Ultimately, our work paves the way for AI systems that can truly understand, appreciate, and generate artworks that align with the sensible human aesthetic standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v3">Unlocking LLM Repair Capabilities in Low-Resource Programming Languages Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v2">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13052v1">GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20973v2">Arabizi vs LLMs: Can the Genie Understand the Language of Aladdin?</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted to MT Summit 2025 (Track: Implementation and Case Studies) https://mtsummit2025.unige.ch/
    </div>
    <details class="paper-abstract">
      In this era of rapid technological advancements, communication continues to evolve as new linguistic phenomena emerge. Among these is Arabizi, a hybrid form of Arabic that incorporates Latin characters and numbers to represent the spoken dialects of Arab communities. Arabizi is widely used on social media and allows people to communicate in an informal and dynamic way, but it poses significant challenges for machine translation due to its lack of formal structure and deeply embedded cultural nuances. This case study arises from a growing need to translate Arabizi for gisting purposes. It evaluates the capacity of different LLMs to decode and translate Arabizi, focusing on multiple Arabic dialects that have rarely been studied up until now. Using a combination of human evaluators and automatic metrics, this research project investigates the model's performance in translating Arabizi into both Modern Standard Arabic and English. Key questions explored include which dialects are translated most effectively and whether translations into English surpass those into Arabic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13038v1">How Large Language Models Are Changing MOOC Essay Answers: A Comparison of Pre- and Post-LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The release of ChatGPT in late 2022 caused a flurry of activity and concern in the academic and educational communities. Some see the tool's ability to generate human-like text that passes at least cursory inspections for factual accuracy ``often enough'' a golden age of information retrieval and computer-assisted learning. Some, on the other hand, worry the tool may lead to unprecedented levels of academic dishonesty and cheating. In this work, we quantify some of the effects of the emergence of Large Language Models (LLMs) on online education by analyzing a multi-year dataset of student essay responses from a free university-level MOOC on AI ethics. Our dataset includes essays submitted both before and after ChatGPT's release. We find that the launch of ChatGPT coincided with significant changes in both the length and style of student essays, mirroring observations in other contexts such as academic publishing. We also observe -- as expected based on related public discourse -- changes in prevalence of key content words related to AI and LLMs, but not necessarily the general themes or topics discussed in the student essays as identified through (dynamic) topic modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13032v1">InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 This paper has been accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their use as agents for planning complex tasks. Existing methods typically rely on a thought-action-observation (TAO) process to enhance LLM performance, but these approaches are often constrained by the LLMs' limited knowledge of complex tasks. Retrieval-augmented generation (RAG) offers new opportunities by leveraging external databases to ground generation in retrieved information. In this paper, we identify two key challenges (enlargability and transferability) in applying RAG to task planning. We propose InstructRAG, a novel solution within a multi-agent meta-reinforcement learning framework, to address these challenges. InstructRAG includes a graph to organize past instruction paths (sequences of correct actions), an RL-Agent with Reinforcement Learning to expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to improve task generalization for transferability. The two agents are trained end-to-end to optimize overall planning performance. Our experiments on four widely used task planning datasets demonstrate that InstructRAG significantly enhances performance and adapts efficiently to new tasks, achieving up to a 19.2% improvement over the best existing approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11672v3">MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      While current large language models (LLMs) perform well on many knowledge-related tasks, they are limited by relying on their parameters as an implicit storage mechanism. As a result, they struggle with memorizing rare events and with updating their memory as facts change over time. In addition, the uninterpretable nature of parametric memory makes it challenging to prevent hallucination. Model editing and augmenting LLMs with parameters specialized for memory are only partial solutions. In this paper, we introduce MemLLM, a novel method of enhancing LLMs by integrating a structured and explicit read-and-write memory module. MemLLM tackles the aforementioned challenges by enabling dynamic interaction with the memory and improving the LLM's capabilities in using stored knowledge. Our experiments indicate that MemLLM enhances the LLM's performance and interpretability, in language modeling in general and knowledge-intensive tasks in particular. We see MemLLM as an important step towards making LLMs more grounded and factual through memory augmentation. The project repository is publicly available at https://github.com/amodaresi/MemLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12984v1">A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) is critical for AI-powered applications but demands substantial computational resources, particularly in memory bandwidth and computational throughput. Low-precision computation has emerged as a key technique to improve efficiency while reducing resource consumption. Existing approaches for generating low-precision kernels are limited to weight bit widths that are powers of two and suffer from suboptimal performance due to high-level GPU programming abstractions. These abstractions restrict critical optimizations, such as fine-grained register management and optimized memory access patterns, which are essential for efficient low-precision computations. In this paper, we introduce a virtual machine (VM) designed for General-Purpose GPU (GPGPU) computing, enabling support for low-precision data types with arbitrary bit widths while maintaining GPU programmability. The proposed VM features a thread-block-level programming model, a hierarchical memory space, a novel algebraic layout system, and extensive support for diverse low-precision data types. VM programs are compiled into highly efficient GPU programs with automatic vectorization and instruction selection. Extensive experiments demonstrate that our VM efficiently supports a full spectrum of low-precision data types, and outperforms state-of-the-art low-precision kernels on their supported types. Compared to existing compilers like Triton and Ladder, as well as hand-optimized kernels such as QuantLLM and Marlin, our VM achieves performance improvements of 1.75x, 2.61x, 1.29x and 1.03x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12982v1">Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) has significantly advanced information retrieval systems, particularly in response generation (RG). Unfortunately, LLMs often face knowledge conflicts between internal memory and retrievaled external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences. However, when the distinction is ambiguous, LLMs experience heightened uncertainty. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models into adaptive augmentation of retrieved information and guiding LLM preference in response generation. Extensive experiments on single-choice, open-ended question-answering (QA), and retrieval augmented generation (RAG) validate our theoretical findings and demonstrate the efficacy of Swin-VIB. Notably, our method improves single-choice task accuracy by at least 7.54\% over competitive baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12943v1">Customizing Emotional Support: How Do Individuals Construct and Interact With LLM-Powered Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 20 pages, 3 figures, 3 tables. Accepted to CHI 2025, ACM Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Personalized support is essential to fulfill individuals' emotional needs and sustain their mental well-being. Large language models (LLMs), with great customization flexibility, hold promises to enable individuals to create their own emotional support agents. In this work, we developed ChatLab, where users could construct LLM-powered chatbots with additional interaction features including voices and avatars. Using a Research through Design approach, we conducted a week-long field study followed by interviews and design activities (N = 22), which uncovered how participants created diverse chatbot personas for emotional reliance, confronting stressors, connecting to intellectual discourse, reflecting mirrored selves, etc. We found that participants actively enriched the personas they constructed, shaping the dynamics between themselves and the chatbot to foster open and honest conversations. They also suggested other customizable features, such as integrating online activities and adjustable memory settings. Based on these findings, we discuss opportunities for enhancing personalized emotional support through emerging AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12867v1">EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and modality-of-thought (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at https://anonymous.4open.science/r/EmoVoice-DF55. Dataset, code, and checkpoints will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12865v1">DashChat: Interactive Authoring of Industrial Dashboard Design Prototypes through Conversation with LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Industrial dashboards, commonly deployed by organizations such as enterprises and governments, are increasingly crucial in data communication and decision-making support across various domains. Designing an industrial dashboard prototype is particularly challenging due to its visual complexity, which can include data visualization, layout configuration, embellishments, and animations. Additionally, in real-world industrial settings, designers often encounter numerous constraints. For instance, when companies negotiate collaborations with clients and determine design plans, they typically need to demo design prototypes and iterate on them based on mock data quickly. Such a task is very common and crucial during the ideation stage, as it not only helps save developmental costs but also avoids data-related issues such as lengthy data handover periods. However, existing authoring tools of dashboards are mostly not tailored to such prototyping needs, and motivated by these gaps, we propose DashChat, an interactive system that leverages large language models (LLMs) to generate industrial dashboard design prototypes from natural language. We collaborated closely with designers from the industry and derived the requirements based on their practical experience. First, by analyzing 114 high-quality industrial dashboards, we summarized their common design patterns and inject the identified ones into LLMs as reference. Next, we built a multi-agent pipeline powered by LLMs to understand textual requirements from users and generate practical, aesthetic prototypes. Besides, functionally distinct, parallel-operating agents are created to enable efficient generation. Then, we developed a user-friendly interface that supports text-based interaction for generating and modifying prototypes. Two user studies demonstrated that our system is both effective and efficient in supporting design prototyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12845v1">Can LLMs reason over extended multilingual contexts? Towards long-context evaluation beyond retrieval and haystacks</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 33 Pages in Total - 23 (Main Manuscript) + 10 (Appendix)
    </div>
    <details class="paper-abstract">
      Existing multilingual long-context benchmarks, often based on the popular needle-in-a-haystack test, primarily evaluate a model's ability to locate specific information buried within irrelevant texts. However, such a retrieval-centric approach is myopic and inherently limited, as successful recall alone does not indicate a model's capacity to reason over extended contexts. Moreover, these benchmarks are susceptible to data leakage, short-circuiting, and risk making the evaluation a priori identifiable. To address these limitations, we introduce MLRBench, a new synthetic benchmark for multilingual long-context reasoning. Unlike existing benchmarks, MLRBench goes beyond surface-level retrieval by including tasks that assess multi-hop inference, aggregation, and epistemic reasoning. Spanning seven languages, MLRBench is designed to be parallel, resistant to leakage, and scalable to arbitrary context lengths. Our extensive experiments with an open-weight large language model (LLM) reveal a pronounced gap between high- and low-resource languages, particularly for tasks requiring the model to aggregate multiple facts or predict the absence of information. We also find that, in multilingual settings, LLMs effectively utilize less than 30% of their claimed context length. Although off-the-shelf Retrieval Augmented Generation helps alleviate this to a certain extent, it does not solve the long-context problem. We open-source MLRBench to enable future research in improved evaluation and training of multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12805v1">Assesing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 30 pages, 13 figures, 1 table
    </div>
    <details class="paper-abstract">
      This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12773v1">Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Multimodal Large Language Models (MLLMs) have achieved remarkable progress in general domains and demonstrated promise in multimodal mathematical reasoning. However, applying MLLMs to geometry problem solving (GPS) remains challenging due to lack of accurate step-by-step solution data and severe hallucinations during reasoning. In this paper, we propose GeoGen, a pipeline that can automatically generates step-wise reasoning paths for geometry diagrams. By leveraging the precise symbolic reasoning, \textbf{GeoGen} produces large-scale, high-quality question-answer pairs. To further enhance the logical reasoning ability of MLLMs, we train \textbf{GeoLogic}, a Large Language Model (LLM) using synthetic data generated by GeoGen. Serving as a bridge between natural language and symbolic systems, GeoLogic enables symbolic tools to help verifying MLLM outputs, making the reasoning process more rigorous and alleviating hallucinations. Experimental results show that our approach consistently improves the performance of MLLMs, achieving remarkable results on benchmarks for geometric reasoning tasks. This improvement stems from our integration of the strengths of LLMs and symbolic systems, which enables a more reliable and interpretable approach for the GPS task. Codes are available at https://github.com/ycpNotFound/GeoGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12732v1">Validating LLM-Generated Relevance Labels for Educational Resource Search</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Presented in the LLM4Eval Workshop Co-located with WSDM '25 in Hannover, Germany
    </div>
    <details class="paper-abstract">
      Manual relevance judgements in Information Retrieval are costly and require expertise, driving interest in using Large Language Models (LLMs) for automatic assessment. While LLMs have shown promise in general web search scenarios, their effectiveness for evaluating domain-specific search results, such as educational resources, remains unexplored. To investigate different ways of including domain-specific criteria in LLM prompts for relevance judgement, we collected and released a dataset of 401 human relevance judgements from a user study involving teaching professionals performing search tasks related to lesson planning. We compared three approaches to structuring these prompts: a simple two-aspect evaluation baseline from prior work on using LLMs as relevance judges, a comprehensive 12-dimensional rubric derived from educational literature, and criteria directly informed by the study participants. Using domain-specific frameworks, LLMs achieved strong agreement with human judgements (Cohen's $\kappa$ up to 0.650), significantly outperforming the baseline approach. The participant-derived framework proved particularly robust, with GPT-3.5 achieving $\kappa$ scores of 0.639 and 0.613 for 10-dimension and 5-dimension versions respectively. System-level evaluation showed that LLM judgements reliably identified top-performing retrieval approaches (RBO scores 0.71-0.76) while maintaining reasonable discrimination between systems (RBO 0.52-0.56). These findings suggest that LLMs can effectively evaluate educational resources when prompted with domain-specific criteria, though performance varies with framework complexity and input structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24047v2">Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 34 pages, 10 figures
    </div>
    <details class="paper-abstract">
      As scientific research becomes increasingly complex, innovative tools are needed to manage vast data, facilitate interdisciplinary collaboration, and accelerate discovery. Large language models (LLMs) are now evolving into LLM-based scientific agents that automate critical tasks, ranging from hypothesis generation and experiment design to data analysis and simulation. Unlike general-purpose LLMs, these specialized agents integrate domain-specific knowledge, advanced tool sets, and robust validation mechanisms, enabling them to handle complex data types, ensure reproducibility, and drive scientific breakthroughs. This survey provides a focused review of the architectures, design, benchmarks, applications, and ethical considerations surrounding LLM-based scientific agents. We highlight why they differ from general agents and the ways in which they advance research across various scientific fields. By examining their development and challenges, this survey offers a comprehensive roadmap for researchers and practitioners to harness these agents for more efficient, reliable, and ethically sound scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12691v1">Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate hallucinations-content that deviates from factual accuracy or provided context-posing challenges for diagnosis due to the complex interplay of underlying causes. This paper introduces a subsequence association framework to systematically trace and understand hallucinations. Our key insight is that hallucinations arise when dominant hallucinatory associations outweigh faithful ones. Through theoretical and empirical analyses, we demonstrate that decoder-only transformers effectively function as subsequence embedding models, with linear layers encoding input-output associations. We propose a tracing algorithm that identifies causal subsequences by analyzing hallucination probabilities across randomized input contexts. Experiments show our method outperforms standard attribution techniques in identifying hallucination causes and aligns with evidence from the model's training corpus. This work provides a unified perspective on hallucinations and a robust framework for their tracing and analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12687v1">Data-efficient LLM Fine-tuning for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 arXiv admin note: text overlap with arXiv:2408.02193
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in code generation tasks. However, there remains a performance gap between open-source and closed-source models. To address this gap, existing approaches typically generate large amounts of synthetic data for fine-tuning, which often leads to inefficient training. In this work, we propose a data selection strategy in order to improve the effectiveness and efficiency of training for code-based LLMs. By prioritizing data complexity and ensuring that the sampled subset aligns with the distribution of the original dataset, our sampling strategy effectively selects high-quality data. Additionally, we optimize the tokenization process through a "dynamic pack" technique, which minimizes padding tokens and reduces computational resource consumption. Experimental results show that when training on 40% of the OSS-Instruct dataset, the DeepSeek-Coder-Base-6.7B model achieves an average performance of 66.9%, surpassing the 66.1% performance with the full dataset. Moreover, training time is reduced from 47 minutes to 34 minutes, and the peak GPU memory decreases from 61.47 GB to 42.72 GB during a single epoch. Similar improvements are observed with the CodeLlama-Python-7B model on the Evol-Instruct dataset. By optimizing both data selection and tokenization, our approach not only improves model performance but also improves training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12682v1">WebLists: Extracting Structured Information From Complex Interactive Websites Using Executable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Most recent web agent research has focused on navigation and transaction tasks, with little emphasis on extracting structured data at scale. We present WebLists, a benchmark of 200 data-extraction tasks across four common business and enterprise use-cases. Each task requires an agent to navigate to a webpage, configure it appropriately, and extract complete datasets with well-defined schemas. We show that both LLMs with search capabilities and SOTA web agents struggle with these tasks, with a recall of 3% and 31%, respectively, despite higher performance on question-answering tasks. To address this challenge, we propose BardeenAgent, a novel framework that enables web agents to convert their execution into repeatable programs, and replay them at scale across pages with similar structure. BardeenAgent is also the first LLM agent to take advantage of the regular structure of HTML. In particular BardeenAgent constructs a generalizable CSS selector to capture all relevant items on the page, then fits the operations to extract the data. On the WebLists benchmark, BardeenAgent achieves 66% recall overall, more than doubling the performance of SOTA web agents, and reducing cost per output row by 3x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12681v1">GRAIL: Gradient-Based Adaptive Unlearning for Privacy and Copyright in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted by IJCNN 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on extensive datasets often learn sensitive information, which raises significant social and legal concerns under principles such as the "Right to be forgotten." Retraining entire models from scratch to remove undesired information is both costly and impractical. Furthermore, existing single-domain unlearning methods fail to address multi-domain scenarios, where knowledge is interwoven across domains such as privacy and copyright, creating overlapping representations that lead to excessive knowledge removal or degraded performance. To tackle these issues, we propose GRAIL (GRadient-based AdaptIve unLearning), a novel multi-domain unlearning framework. GRAIL leverages gradient information from multiple domains to precisely distinguish the unlearning scope from the retention scope, and applies an adaptive parameter-wise localization strategy to selectively remove targeted knowledge while preserving critical parameters for each domain. Experimental results on unlearning benchmarks show that GRAIL achieves unlearning success on par with the existing approaches, while also demonstrating up to 17% stronger knowledge retention success compared to the previous state-of-art method. Our findings establish a new paradigm for effectively managing and regulating sensitive information in large-scale pre-trained language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12637v1">Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 26 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) struggle with long-context reasoning, not only due to the quadratic scaling of computational complexity with sequence length but also because of the scarcity and expense of annotating long-context data. There has been barely any open-source work that systematically ablates long-context data, nor is there any openly available instruction tuning dataset with contexts surpassing 100K tokens. To bridge this gap, we introduce a novel post-training synthetic data generation strategy designed to efficiently extend the context window of LLMs while preserving their general task performance. Our approach scalably extends to arbitrarily long context lengths, unconstrained by the length of available real-world data, which effectively addresses the scarcity of raw long-context data. Through a step-by-step rotary position embedding (RoPE) scaling training strategy, we demonstrate that our model, with a context length of up to 1M tokens, performs well on the RULER benchmark and InfiniteBench and maintains robust performance on general language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12876v3">In-context KV-Cache Eviction for LLMs via Attention-Gate</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      The KV-Cache technique has become the standard for the inference of large language models (LLMs). Yet, it is widely criticized that KV-Cache can become a bottleneck of the LLM inference system. This paper enables a novel dynamic KV-Cache eviction policy by injecting a lightweight module called Attention-Gate to the model. It accepts the global context as input and yields eviction flags for each token. The self-attention modules in the model proceed according to the flags and cache only a subset of the KV states for next token prediction. The Attention-Gates can yield various flags for different heads and layers and be easily tuned on top of a pre-trained LLM via continual pre-training or supervised fine-tuning. The computational and memory overhead introduced by Attention-Gates can be minimal. We empirically evaluate the proposed approach across multiple scenarios, showing that effective eviction of redundant tokens can not only improve efficiency but also enhance performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09593v2">ControlNET: A Firewall for RAG-based LLM System</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Project Page: https://ai.zjuicsr.cn/firewall
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15395v2">Beyond Tools: Understanding How Heavy Users Integrate LLMs into Everyday Tasks and Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for both everyday and specialized tasks. While HCI research focuses on domain-specific applications, little is known about how heavy users integrate LLMs into everyday decision-making. Through qualitative interviews with heavy LLM users (n=7) who employ these systems for both intuitive and analytical thinking tasks, our findings show that participants use LLMs for social validation, self-regulation, and interpersonal guidance, seeking to build self-confidence and optimize cognitive resources. These users viewed LLMs either as rational, consistent entities or average human decision-makers. Our findings suggest that heavy LLM users develop nuanced interaction patterns beyond simple delegation, highlighting the need to reconsider how we study LLM integration in decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v3">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12608v1">Code Copycat Conundrum: Demystifying Repetition in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Despite recent advances in Large Language Models (LLMs) for code generation, the quality of LLM-generated code still faces significant challenges. One significant issue is code repetition, which refers to the model's tendency to generate structurally redundant code, resulting in inefficiencies and reduced readability. To address this, we conduct the first empirical study to investigate the prevalence and nature of repetition across 19 state-of-the-art code LLMs using three widely-used benchmarks. Our study includes both quantitative and qualitative analyses, revealing that repetition is pervasive and manifests at various granularities and extents, including character, statement, and block levels. We further summarize a taxonomy of 20 repetition patterns. Building on our findings, we propose DeRep, a rule-based technique designed to detect and mitigate repetition in generated code. We evaluate DeRep using both open-source benchmarks and in an industrial setting. Our results demonstrate that DeRep significantly outperforms baselines in reducing repetition (with an average improvements of 91.3%, 93.5%, and 79.9% in rep-3, rep-line, and sim-line metrics) and enhancing code quality (with a Pass@1 increase of 208.3% over greedy search). Furthermore, integrating DeRep improves the performance of existing repetition mitigation methods, with Pass@1 improvements ranging from 53.7% to 215.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07791v8">Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has emerged as a promising alternative to human evaluators across various tasks, yet inherent biases - particularly position bias, the tendency to favor solutions based on their position within the prompt - compromise its reliability. This exploratory study evaluates position bias in LLM judges across pairwise and list-wise comparison settings, introducing three metrics: repetition stability, position consistency, and preference fairness. Our experiments, involving 15 LLM judges across MTBench and DevBench with 22 tasks and approximately 40 solution-generating models, result in over 150,000 evaluation instances. We identify Judge-Level, Candidate-Level, and Task-Level factors contributing to bias. The findings confirm that position bias is not due to random chance and varies significantly across judges and tasks. While position bias is weakly influenced by the length of prompt components, it is strongly affected by the quality gap between solutions. Our agreement and disagreement analysis among judges further provides insights into the distribution of judging difficulty across the dataset, and highlights the potential for dataset modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11711v2">The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Static analysis is a cornerstone for software vulnerability detection, yet it often struggles with the classic precision-scalability trade-off. In practice, such tools often produce high false positive rates, particularly in large codebases like the Linux kernel. This imprecision can arise from simplified vulnerability modeling and over-approximation of path and data constraints. While large language models (LLMs) show promise in code understanding, their naive application to program analysis yields unreliable results due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly improves static analysis precision. BugLens guides an LLM to follow traditional analysis steps by assessing buggy code patterns for security impact and validating the constraints associated with static warnings. Evaluated on real-world Linux kernel bugs, BugLens raises precision from 0.10 (raw) and 0.50 (semi-automated refinement) to 0.72, substantially reducing false positives and revealing four previously unreported vulnerabilities. Our results suggest that a structured LLM-based workflow can meaningfully enhance the effectiveness of static analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10673v2">ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      We introduce ZeroSumEval, a dynamic, competition-based, and evolving evaluation framework for Large Language Models (LLMs) that leverages competitive games. ZeroSumEval encompasses a diverse suite of games, including security challenges (Capture the Flag), classic board games (chess), and knowledge tests (MathQuiz). These games are designed to evaluate a range of capabilities such as strategic reasoning, planning, knowledge application, safety, and adaptability. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework for easily implementing games and leverages DSPy to provide a better abstraction for LLM player strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12589v1">Efficient MAP Estimation of LLM Judgment Performance with Prior Transfer</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      LLM ensembles are widely used for LLM judges. However, how to estimate their accuracy, especially in an efficient way, is unknown. In this paper, we present a principled maximum a posteriori (MAP) framework for an economical and precise estimation of the performance of LLM ensemble judgment. We first propose a mixture of Beta-Binomial distributions to model the judgment distribution, revising from the vanilla Binomial distribution. Next, we introduce a conformal prediction-driven approach that enables adaptive stopping during iterative sampling to balance accuracy with efficiency. Furthermore, we design a prior transfer mechanism that utilizes learned distributions on open-source datasets to improve estimation on a target dataset when only scarce annotations are available. Finally, we present BetaConform, a framework that integrates our distribution assumption, adaptive stopping, and the prior transfer mechanism to deliver a theoretically guaranteed distribution estimation of LLM ensemble judgment with minimum labeled samples. BetaConform is also validated empirically. For instance, with only 10 samples from the TruthfulQA dataset, for a Llama ensembled judge, BetaConform gauges its performance with error margin as small as 3.37%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08260v2">Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09956v3">DeepSeek-Inspired Exploration of RL-based LLMs and Synergy with Wireless Networks: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 45 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have gained significant attention for their exceptional capabilities in natural language processing and multimodal data understanding. Meanwhile, the rapid expansion of information services has driven the growing need for intelligence, efficient, and adaptable wireless networks. Wireless networks require the empowerment of RL-based LLMs while these models also benefit from wireless networks to broaden their application scenarios. Specifically, RL-based LLMs can enhance wireless communication systems through intelligent resource allocation, adaptive network optimization, and real-time decision-making. Conversely, wireless networks provide a vital infrastructure for the efficient training, deployment, and distributed inference of RL-based LLMs, especially in decentralized and edge computing environments. This mutual empowerment highlights the need for a deeper exploration of the interplay between these two domains. We first review recent advancements in wireless communications, highlighting the associated challenges and potential solutions. We then discuss the progress of RL-based LLMs, focusing on key technologies for LLM training, challenges, and potential solutions. Subsequently, we explore the mutual empowerment between these two fields, highlighting key motivations, open challenges, and potential solutions. Finally, we provide insights into future directions, applications, and their societal impact to further explore this intersection, paving the way for next-generation intelligent communication systems. Overall, this survey provides a comprehensive overview of the relationship between RL-based LLMs and wireless networks, offering a vision where these domains empower each other to drive innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12562v1">ZeroSumEval: Scaling LLM Evaluation with Inter-Model Competition</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Evaluating the capabilities of Large Language Models (LLMs) has traditionally relied on static benchmark datasets, human assessments, or model-based evaluations - methods that often suffer from overfitting, high costs, and biases. ZeroSumEval is a novel competition-based evaluation protocol that leverages zero-sum games to assess LLMs with dynamic benchmarks that resist saturation. ZeroSumEval encompasses a diverse suite of games, including security challenges (PyJail), classic games (Chess, Liar's Dice, Poker), knowledge tests (MathQuiz), and persuasion challenges (Gandalf, Debate). These games are designed to evaluate a range of AI capabilities such as strategic reasoning, planning, knowledge application, and creativity. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework. To demonstrate this, we conduct extensive experiments with >7000 simulations across 7 games and 13 models. Our results show that while frontier models from the GPT and Claude families can play common games and answer questions, they struggle to play games that require creating novel and challenging questions. We also observe that models cannot reliably jailbreak each other and fail generally at tasks requiring creativity. We release our code at https://github.com/facebookresearch/ZeroSumEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12558v1">Benchmarking LLM-based Relevance Judgment Methods</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in both academic and industry settings to automate the evaluation of information seeking systems, particularly by generating graded relevance judgments. Previous work on LLM-based relevance assessment has primarily focused on replicating graded human relevance judgments through various prompting strategies. However, there has been limited exploration of alternative assessment methods or comprehensive comparative studies. In this paper, we systematically compare multiple LLM-based relevance assessment methods, including binary relevance judgments, graded relevance assessments, pairwise preference-based methods, and two nugget-based evaluation methods~--~document-agnostic and document-dependent. In addition to a traditional comparison based on system rankings using Kendall correlations, we also examine how well LLM judgments align with human preferences, as inferred from relevance grades. We conduct extensive experiments on datasets from three TREC Deep Learning tracks 2019, 2020 and 2021 as well as the ANTIQUE dataset, which focuses on non-factoid open-domain question answering. As part of our data release, we include relevance judgments generated by both an open-source (Llama3.2b) and a commercial (gpt-4o) model. Our goal is to \textit{reproduce} various LLM-based relevance judgment methods to provide a comprehensive comparison. All code, data, and resources are publicly available in our GitHub Repository at https://github.com/Narabzad/llm-relevance-judgement-comparison.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12553v1">ELAB: Extensive LLM Alignment Benchmark in Persian Language</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive evaluation framework for aligning Persian Large Language Models (LLMs) with critical ethical dimensions, including safety, fairness, and social norms. It addresses the gaps in existing LLM evaluation frameworks by adapting them to Persian linguistic and cultural contexts. This benchmark creates three types of Persian-language benchmarks: (i) translated data, (ii) new data generated synthetically, and (iii) new naturally collected data. We translate Anthropic Red Teaming data, AdvBench, HarmBench, and DecodingTrust into Persian. Furthermore, we create ProhibiBench-fa, SafeBench-fa, FairBench-fa, and SocialBench-fa as new datasets to address harmful and prohibited content in indigenous culture. Moreover, we collect extensive dataset as GuardBench-fa to consider Persian cultural norms. By combining these datasets, our work establishes a unified framework for evaluating Persian LLMs, offering a new approach to culturally grounded alignment evaluation. A systematic evaluation of Persian LLMs is performed across the three alignment aspects: safety (avoiding harmful content), fairness (mitigating biases), and social norms (adhering to culturally accepted behaviors). We present a publicly available leaderboard that benchmarks Persian LLMs with respect to safety, fairness, and social norms at: https://huggingface.co/spaces/MCILAB/LLM_Alignment_Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12427v1">Position: The Most Expensive Part of an LLM should be its Training Data</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Training a state-of-the-art Large Language Model (LLM) is an increasingly expensive endeavor due to growing computational, hardware, energy, and engineering demands. Yet, an often-overlooked (and seldom paid) expense is the human labor behind these models' training data. Every LLM is built on an unfathomable amount of human effort: trillions of carefully written words sourced from books, academic papers, codebases, social media, and more. This position paper aims to assign a monetary value to this labor and argues that the most expensive part of producing an LLM should be the compensation provided to training data producers for their work. To support this position, we study 64 LLMs released between 2016 and 2024, estimating what it would cost to pay people to produce their training datasets from scratch. Even under highly conservative estimates of wage rates, the costs of these models' training datasets are 10-1000 times larger than the costs to train the models themselves, representing a significant financial liability for LLM providers. In the face of the massive gap between the value of training data and the lack of compensation for its creation, we highlight and discuss research directions that could enable fairer practices in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12422v1">Mitigating LLM Hallucinations with Knowledge Graphs: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Presented at the Human-centered Explainable AI Workshop (HCXAI) @ CHI 2025
    </div>
    <details class="paper-abstract">
      High-stakes domains like cyber operations need responsible and trustworthy AI methods. While large language models (LLMs) are becoming increasingly popular in these domains, they still suffer from hallucinations. This research paper provides learning outcomes from a case study with LinkQ, an open-source natural language interface that was developed to combat hallucinations by forcing an LLM to query a knowledge graph (KG) for ground-truth data during question-answering (QA). We conduct a quantitative evaluation of LinkQ using a well-known KGQA dataset, showing that the system outperforms GPT-4 but still struggles with certain question categories - suggesting that alternative query construction strategies will need to be investigated in future LLM querying systems. We discuss a qualitative study of LinkQ with two domain experts using a real-world cybersecurity KG, outlining these experts' feedback, suggestions, perceived limitations, and future opportunities for systems like LinkQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12408v1">A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate relevance judgments for information retrieval (IR) tasks, often demonstrating agreement with human labels that approaches inter-human agreement. To assess the robustness and reliability of LLM-based relevance judgments, we systematically investigate impact of prompt sensitivity on the task. We collected prompts for relevance assessment from 15 human experts and 15 LLMs across three tasks~ -- ~binary, graded, and pairwise~ -- ~yielding 90 prompts in total. After filtering out unusable prompts from three humans and three LLMs, we employed the remaining 72 prompts with three different LLMs as judges to label document/query pairs from two TREC Deep Learning Datasets (2020 and 2021). We compare LLM-generated labels with TREC official human labels using Cohen's $\kappa$ and pairwise agreement measures. In addition to investigating the impact of prompt variations on agreement with human labels, we compare human- and LLM-generated prompts and analyze differences among different LLMs as judges. We also compare human- and LLM-generated prompts with the standard UMBRELA prompt used for relevance assessment by Bing and TREC 2024 Retrieval Augmented Generation (RAG) Track. To support future research in LLM-based evaluation, we release all data and prompts at https://github.com/Narabzad/prompt-sensitivity-relevance-judgements/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12397v1">Activated LoRA: Fine-tuned LLMs for Intrinsics</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 arXiv admin note: text overlap with arXiv:2504.11704
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has emerged as a highly efficient framework for finetuning the weights of large foundation models, and has become the go-to method for data-driven customization of LLMs. Despite the promise of highly customized behaviors and capabilities, switching between relevant LoRAs in a multiturn setting is highly inefficient, as the key-value (KV) cache of the entire turn history must be recomputed with the LoRA weights before generation can begin. To address this problem, we propose Activated LoRA (aLoRA), which modifies the LoRA framework to only adapt weights for the tokens in the sequence \emph{after} the aLoRA is invoked. This change crucially allows aLoRA to accept the base model's KV cache of the input string, meaning that aLoRA can be instantly activated whenever needed in a chain without recomputing the cache. This enables building what we call \emph{intrinsics}, i.e. highly specialized models invoked to perform well-defined operations on portions of an input chain or conversation that otherwise uses the base model by default. We use aLoRA to train a set of intrinsics models, demonstrating competitive accuracy with standard LoRA while achieving significant inference benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07447v3">Optimizing LLM Inference for Database Systems: Cost-Aware Scheduling for Concurrent Requests</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used inside database systems and in database applications for better complexity management and decision-making, where LLM inferences require significant GPU costs. LLM inference systems, however, are slow compared to database systems, limiting the expansion of the use of LLMs inside database systems. This paper first analyzes the LLM inference performance and focuses on a data management issue in LLM inference. We reveal that the root of the problem is the lack of an adequate resource cost model and optimization strategy when executing multiple concurrent inference requests. We adapt classic database multi-query optimization techniques by introducing cost models for concurrent inference requests and new scheduling strategies to optimize the use of memory resources by concurrent requests, thereby substantially improving performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13209v1">On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02108v2">ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12268v1">HLS-Eval: A Benchmark and Framework for Evaluating LLMs on High-Level Synthesis Design Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The rapid scaling of large language model (LLM) training and inference has driven their adoption in semiconductor design across academia and industry. While most prior work evaluates LLMs on hardware description language (HDL) tasks, particularly Verilog, designers are increasingly using high-level synthesis (HLS) to build domain-specific accelerators and complex hardware systems. However, benchmarks and tooling to comprehensively evaluate LLMs for HLS design tasks remain scarce. To address this, we introduce HLS-Eval, the first complete benchmark and evaluation framework for LLM-driven HLS design. HLS-Eval targets two core tasks: (1) generating HLS code from natural language descriptions, and (2) performing HLS-specific code edits to optimize performance and hardware efficiency. The benchmark includes 94 unique designs drawn from standard HLS benchmarks and novel sources. Each case is prepared via a semi-automated flow that produces a natural language description and a paired testbench for C-simulation and synthesis validation, ensuring each task is "LLM-ready." Beyond the benchmark, HLS-Eval offers a modular Python framework for automated, parallel evaluation of both local and hosted LLMs. It includes a parallel evaluation engine, direct HLS tool integration, and abstractions for to support different LLM interaction paradigms, enabling rapid prototyping of new benchmarks, tasks, and LLM methods. We demonstrate HLS-Eval through baseline evaluations of open-source LLMs on Vitis HLS, measuring outputs across four key metrics - parseability, compilability, runnability, and synthesizability - reflecting the iterative HLS design cycle. We also report pass@k metrics, establishing clear baselines and reusable infrastructure for the broader LLM-for-hardware community. All benchmarks, framework code, and results are open-sourced at https://github.com/stefanpie/hls-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12250v1">AnomalyGen: An Automated Semantic Log Sequence Generation Framework with LLM for Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The scarcity of high-quality public log datasets has become a critical bottleneck in advancing log-based anomaly detection techniques. Current datasets exhibit three fundamental limitations: (1) incomplete event coverage, (2) artificial patterns introduced by static analysis-based generation frameworks, and (3) insufficient semantic awareness. To address these challenges, we present AnomalyGen, the first automated log synthesis framework specifically designed for anomaly detection. Our framework introduces a novel four-phase architecture that integrates enhanced program analysis with Chain-of-Thought reasoning (CoT reasoning), enabling iterative log generation and anomaly annotation without requiring physical system execution. Evaluations on Hadoop and HDFS distributed systems demonstrate that AnomalyGen achieves substantially broader log event coverage (38-95 times improvement over existing datasets) while producing more operationally realistic log sequences compared to static analysis-based approaches. When augmenting benchmark datasets with synthesized logs, we observe maximum F1-score improvements of 3.7% (average 1.8% improvement across three state-of-the-art anomaly detection models). This work not only establishes a high-quality benchmarking resource for automated log analysis but also pioneers a new paradigm for applying large language models (LLMs) in software engineering workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12185v1">SALAD: Improving Robustness and Generalization through Contrastive Learning with Structure-Aware and LLM-Driven Augmented Data</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted to NAACL 2025 main. 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In various natural language processing (NLP) tasks, fine-tuning Pre-trained Language Models (PLMs) often leads to the issue of spurious correlations, which negatively impacts performance, particularly when dealing with out-of-distribution data. To address this problem, we propose SALAD}(Structure Aware and LLM-driven Augmented Data), a novel approach designed to enhance model robustness and generalization by generating structure-aware and counterfactually augmented data for contrastive learning. Our method leverages a tagging-based approach to generate structure-aware positive samples and utilizes large language models (LLMs) to generate counterfactual negative samples with diverse sentence patterns. By applying contrastive learning, SALAD enables the model to focus on learning the structural relationships between key sentence components while minimizing reliance on spurious correlations. We validate our approach through experiments on three tasks: Sentiment Classification, Sexism Detection, and Natural Language Inference. The results demonstrate that SALAD not only improves model robustness and performance across different environments but also enhances generalization to out-of-distribution datasets and cross-domain scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11168v2">Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 12 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04285v3">Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Along with the proliferating research interest in Semantic Communication (SemCom), Joint Source Channel Coding (JSCC) has dominated the attention due to the widely assumed existence in efficiently delivering information semantics. %has emerged as a pivotal area of research, aiming to enhance the efficiency and reliability of information transmission through deep learning-based methods. Nevertheless, this paper challenges the conventional JSCC paradigm, and advocates for adoption of Separate Source Channel Coding (SSCC) to enjoy the underlying more degree of freedom for optimization. We demonstrate that SSCC, after leveraging the strengths of Large Language Model (LLM) for source coding and Error Correction Code Transformer (ECCT) complemented for channel decoding, offers superior performance over JSCC. Our proposed framework also effectively highlights the compatibility challenges between SemCom approaches and digital communication systems, particularly concerning the resource costs associated with the transmission of high precision floating point numbers. Through comprehensive evaluations, we establish that empowered by LLM-based compression and ECCT-enhanced error correction, SSCC remains a viable and effective solution for modern communication systems. In other words, separate source and channel coding is still what we need!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09063v2">Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Trigger Warning: the appendix contains LLM-generated text with violence and harassment
    </div>
    <details class="paper-abstract">
      Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16514v3">VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08525v3">Task Memory Engine (TME): A Structured Memory Framework with Graph-Aware Extensions for Multi-Step LLM Agent Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 14 pages, 5 figures. Preprint prepared for future submission. Includes implementation and token-efficiency analysis. Code at https://github.com/biubiutomato/TME-Agent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. A reference implementation of the core TME components is available at https://github.com/biubiutomato/TME-Agent, including basic examples and structured memory integration. While the current implementation uses a tree-based structure, TME is designed to be graph-aware, supporting reusable substeps, converging task paths, and shared dependencies. This lays the groundwork for future DAG-based memory architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10185v2">LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12110v1">Towards LLM Agents for Earth Observation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at https://iandrover.github.io/UnivEarth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12108v1">Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12098v1">Gauging Overprecision in LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Recently, overconfidence in large language models (LLMs) has garnered considerable attention due to its fundamental importance in quantifying the trustworthiness of LLM generation. However, existing approaches prompt the \textit{black box LLMs} to produce their confidence (\textit{verbalized confidence}), which can be subject to many biases and hallucinations. Inspired by a different aspect of overconfidence in cognitive science called \textit{overprecision}, we designed a framework for its study in black box LLMs. This framework contains three main phases: 1) generation, 2) refinement and 3) evaluation. In the generation phase we prompt the LLM to generate answers to numerical questions in the form of intervals with a certain level of confidence. This confidence level is imposed in the prompt and not required for the LLM to generate as in previous approaches. We use various prompting techniques and use the same prompt multiple times to gauge the effects of randomness in the generation process. In the refinement phase, answers from the previous phase are refined to generate better answers. The LLM answers are evaluated and studied in the evaluation phase to understand its internal workings. This study allowed us to gain various insights into LLM overprecision: 1) LLMs are highly uncalibrated for numerical tasks 2) {\color{blue}there is no correlation between the length of the interval and the imposed confidence level, which can be symptomatic of a a) lack of understanding of the concept of confidence or b) inability to adjust self-confidence by following instructions}, {\color{blue}3)} LLM numerical precision differs depending on the task, scale of answer and prompting technique {\color{blue}4) Refinement of answers doesn't improve precision in most cases}. We believe this study offers new perspectives on LLM overconfidence and serves as a strong baseline for overprecision in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12034v1">OpDiffer: LLM-Assisted Opcode-Level Differential Testing of Ethereum Virtual Machine</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 To appear in ISSTA'25
    </div>
    <details class="paper-abstract">
      As Ethereum continues to thrive, the Ethereum Virtual Machine (EVM) has become the cornerstone powering tens of millions of active smart contracts. Intuitively, security issues in EVMs could lead to inconsistent behaviors among smart contracts or even denial-of-service of the entire blockchain network. However, to the best of our knowledge, only a limited number of studies focus on the security of EVMs. Moreover, they suffer from 1) insufficient test input diversity and invalid semantics; and 2) the inability to automatically identify bugs and locate root causes. To bridge this gap, we propose OpDiffer, a differential testing framework for EVM, which takes advantage of LLMs and static analysis methods to address the above two limitations. We conducted the largest-scale evaluation, covering nine EVMs and uncovering 26 previously unknown bugs, 22 of which have been confirmed by developers and three have been assigned CNVD IDs. Compared to state-of-the-art baselines, OpDiffer can improve code coverage by at most 71.06%, 148.40% and 655.56%, respectively. Through an analysis of real-world deployed Ethereum contracts, we estimate that 7.21% of the contracts could trigger our identified EVM bugs under certain environmental settings, potentially resulting in severe negative impact on the Ethereum ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11972v1">LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 17 pages; code and data are available at https://github.com/Alab-NII/llm-judge-extract-qa
    </div>
    <details class="paper-abstract">
      Extractive reading comprehension question answering (QA) datasets are typically evaluated using Exact Match (EM) and F1-score, but these metrics often fail to fully capture model performance. With the success of large language models (LLMs), they have been employed in various tasks, including serving as judges (LLM-as-a-judge). In this paper, we reassess the performance of QA models using LLM-as-a-judge across four reading comprehension QA datasets. We examine different families of LLMs and various answer types to evaluate the effectiveness of LLM-as-a-judge in these tasks. Our results show that LLM-as-a-judge is highly correlated with human judgments and can replace traditional EM/F1 metrics. By using LLM-as-a-judge, the correlation with human judgments improves significantly, from 0.17 (EM) and 0.36 (F1-score) to 0.85. These findings confirm that EM and F1 metrics underestimate the true performance of the QA models. While LLM-as-a-judge is not perfect for more difficult answer types (e.g., job), it still outperforms EM/F1, and we observe no bias issues, such as self-preference, when the same model is used for both the QA and judgment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03104v3">ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 accepted by VLDB' 25
    </div>
    <details class="paper-abstract">
      Understanding time series is crucial for its application in real-world scenarios. Recently, large language models (LLMs) have been increasingly applied to time series tasks, leveraging their strong language capabilities to enhance various applications. However, research on multimodal LLMs (MLLMs) for time series understanding and reasoning remains limited, primarily due to the scarcity of high-quality datasets that align time series with textual information. This paper introduces ChatTS, a novel MLLM designed for time series analysis. ChatTS treats time series as a modality, similar to how vision MLLMs process images, enabling it to perform both understanding and reasoning with time series. To address the scarcity of training data, we propose an attribute-based method for generating synthetic time series with detailed attribute descriptions. We further introduce Time Series Evol-Instruct, a novel approach that generates diverse time series Q&As, enhancing the model's reasoning capabilities. To the best of our knowledge, ChatTS is the first TS-MLLM that takes multivariate time series as input for understanding and reasoning, which is fine-tuned exclusively on synthetic datasets. We evaluate its performance using benchmark datasets with real-world data, including six alignment tasks and four reasoning tasks. Our results show that ChatTS significantly outperforms existing vision-based MLLMs (e.g., GPT-4o) and text/agent-based LLMs, achieving a 46.0% improvement in alignment tasks and a 25.8% improvement in reasoning tasks. We have open-sourced the source code, model checkpoint and datasets at https://github.com/NetManAIOps/ChatTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11934v1">An LLM-as-a-judge Approach for Scalable Gender-Neutral Translation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted at GITT 2025
    </div>
    <details class="paper-abstract">
      Gender-neutral translation (GNT) aims to avoid expressing the gender of human referents when the source text lacks explicit cues about the gender of those referents. Evaluating GNT automatically is particularly challenging, with current solutions being limited to monolingual classifiers. Such solutions are not ideal because they do not factor in the source sentence and require dedicated data and fine-tuning to scale to new languages. In this work, we address such limitations by investigating the use of large language models (LLMs) as evaluators of GNT. Specifically, we explore two prompting approaches: one in which LLMs generate sentence-level assessments only, and another, akin to a chain-of-thought approach, where they first produce detailed phrase-level annotations before a sentence-level judgment. Through extensive experiments on multiple languages with five models, both open and proprietary, we show that LLMs can serve as evaluators of GNT. Moreover, we find that prompting for phrase-level annotations before sentence-level assessments consistently improves the accuracy of all models, providing a better and more scalable alternative to current solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11919v1">Rethinking the Generation of High-Quality CoT Data from the Perspective of LLM-Adaptive Question Difficulty Grading</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Recently, DeepSeek-R1 (671B) (DeepSeek-AIet al., 2025) has demonstrated its excellent reasoning ability in complex tasks and has publiclyshared its methodology. This provides potentially high-quality chain-of-thought (CoT) data for stimulating the reasoning abilities of small-sized large language models (LLMs). To generate high-quality CoT data for different LLMs, we seek an efficient method for generating high-quality CoT data with LLM-Adaptive questiondifficulty levels. First, we grade the difficulty of the questions according to the reasoning ability of the LLMs themselves and construct a LLM-Adaptive question database. Second, we sample the problem database based on a distribution of difficulty levels of the questions and then use DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) to generate the corresponding high-quality CoT data with correct answers. Thanks to the construction of CoT data with LLM-Adaptive difficulty levels, we have significantly reduced the cost of data generation and enhanced the efficiency of model supervised fine-tuning (SFT). Finally, we have validated the effectiveness and generalizability of the proposed method in the fields of complex mathematical competitions and code generation tasks. Notably, with only 2k high-quality mathematical CoT data, our ZMath-32B surpasses DeepSeek-Distill-32B in math reasoning task. Similarly, with only 2k high-quality code CoT data, our ZCode-32B surpasses DeepSeek-Distill-32B in code reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11889v1">Rethinking LLM-Based Recommendations: A Query Generation-Based, Training-Free Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Existing large language model LLM-based recommendation methods face several challenges, including inefficiency in handling large candidate pools, sensitivity to item order within prompts ("lost in the middle" phenomenon) poor scalability, and unrealistic evaluation due to random negative sampling. To address these issues, we propose a Query-to-Recommendation approach that leverages LLMs to generate personalized queries for retrieving relevant items from the entire candidate pool, eliminating the need for candidate pre-selection. This method can be integrated into an ID-based recommendation system without additional training, enhances recommendation performance and diversity through LLMs' world knowledge, and performs well even for less popular item groups. Experiments on three datasets show up to 57 percent improvement, with an average gain of 31 percent, demonstrating strong zero-shot performance and further gains when ensembled with existing models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02644v3">Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11833v1">Could Thinking Multilingually Empower LLM Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Previous work indicates that large language models exhibit a significant "English bias", i.e. they often perform better when tasks are presented in English. Interestingly, we have observed that using certain other languages in reasoning tasks can yield better performance than English. However, this phenomenon remains under-explored. In this paper, we explore the upper bound of harnessing multilingualism in reasoning tasks, suggesting that multilingual reasoning promises significantly (by nearly 10 Acc@$k$ points) and robustly (tolerance for variations in translation quality and language choice) higher upper bounds than English-only reasoning. Besides analyzing the reason behind the upper bound and challenges in reaching it, we also find that common answer selection methods cannot achieve this upper bound, due to their limitations and biases. These insights could pave the way for future research aimed at fully harnessing the potential of multilingual reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04820v3">Natural Language Outlines for Code: Literate Programming in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted to FSE'25 Industry Track
    </div>
    <details class="paper-abstract">
      We propose using natural language outlines as a novel modality and interaction surface for providing AI assistance to developers throughout the software development process. An NL outline for a code function comprises multiple statements written in concise prose, which partition the code and summarize its main ideas in the style of literate programming. Crucially, we find that modern LLMs can generate accurate and high-quality NL outlines in practice. Moreover, NL outlines enable a bidirectional sync between code and NL: a developer can change one and the LLM automatically updates the other. We discuss many use cases for NL outlines: they can accelerate understanding and navigation of code and diffs, simplify code maintenance, augment code search, steer code generation, and more. We then propose and compare multiple LLM prompting techniques for generating outlines and ask professional developers to judge outline quality. Finally, we present two case studies applying NL outlines toward code review and malware detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11816v1">Cost-Efficient LLM Serving in the Cloud: VM Selection with KV Cache Offloading</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM inference is essential for applications like text summarization, translation, and data analysis, but the high cost of GPU instances from Cloud Service Providers (CSPs) like AWS is a major burden. This paper proposes InferSave, a cost-efficient VM selection framework for cloud based LLM inference. InferSave optimizes KV cache offloading based on Service Level Objectives (SLOs) and workload charac teristics, estimating GPU memory needs, and recommending cost-effective VM instances. Additionally, the Compute Time Calibration Function (CTCF) improves instance selection accuracy by adjusting for discrepancies between theoretical and actual GPU performance. Experiments on AWS GPU instances show that selecting lower-cost instances without KV cache offloading improves cost efficiency by up to 73.7% for online workloads, while KV cache offloading saves up to 20.19% for offline workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02623v3">Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09080v2">Leveraging Social Determinants of Health in Alzheimer's Research Using LLM-Augmented Literature Mining and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted by AMIA-IS'25: AMIA Informatics Summit
    </div>
    <details class="paper-abstract">
      Growing evidence suggests that social determinants of health (SDoH), a set of nonmedical factors, affect individuals' risks of developing Alzheimer's disease (AD) and related dementias. Nevertheless, the etiological mechanisms underlying such relationships remain largely unclear, mainly due to difficulties in collecting relevant information. This study presents a novel, automated framework that leverages recent advancements of large language model (LLM) and natural language processing techniques to mine SDoH knowledge from extensive literature and integrate it with AD-related biological entities extracted from the general-purpose knowledge graph PrimeKG. Utilizing graph neural networks, we performed link prediction tasks to evaluate the resultant SDoH-augmented knowledge graph. Our framework shows promise for enhancing knowledge discovery in AD and can be generalized to other SDoH-related research areas, offering a new tool for exploring the impact of social determinants on health outcomes. Our code is available at: https://github.com/hwq0726/SDoHenPKG
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11777v1">Bridging the Semantic Gaps: Improving Medical VQA Consistency with LLM-Augmented Question Sets</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 The first two listed authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Medical Visual Question Answering (MVQA) systems can interpret medical images in response to natural language queries. However, linguistic variability in question phrasing often undermines the consistency of these systems. To address this challenge, we propose a Semantically Equivalent Question Augmentation (SEQA) framework, which leverages large language models (LLMs) to generate diverse yet semantically equivalent rephrasings of questions. Specifically, this approach enriches linguistic diversity while preserving semantic meaning. We further introduce an evaluation metric, Total Agreement Rate with Semantically Equivalent Input and Correct Answer (TAR-SC), which assesses a model's capability to generate consistent and correct responses to semantically equivalent linguistic variations. In addition, we also propose three other diversity metrics - average number of QA items per image (ANQI), average number of questions per image with the same answer (ANQA), and average number of open-ended questions per image with the same semantics (ANQS). Using the SEQA framework, we augmented the benchmarked MVQA public datasets of SLAKE, VQA-RAD, and PathVQA. As a result, all three datasets achieved significant improvements by incorporating more semantically equivalent questions: ANQI increased by an average of 86.1, ANQA by 85.1, and ANQS by 46. Subsequent experiments evaluate three MVQA models (M2I2, MUMC, and BiomedGPT) under both zero-shot and fine-tuning settings on the enhanced datasets. Experimental results in MVQA datasets show that fine-tuned models achieve an average accuracy improvement of 19.35%, while our proposed TAR-SC metric shows an average improvement of 11. 61%, indicating a substantial enhancement in model consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09566v2">Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances the reasoning of large language models (LLMs) by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, complex tasks with vast solution spaces and vague constraints often exceed the capacity of a single reasoning chain. Inspired by Minimal Free Resolution (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)-a novel framework that extends CoT by introducing auxiliary, interrelated reasoning paths. SoT captures deeper logical dependencies, enabling more robust and structured problem-solving. MFR decomposes a module into a sequence of free modules with minimal rank, providing a structured analytical approach to complex systems. This method introduces the concepts of "Module", "Betti numbers","Freeness", "Mapping", "Exactness" and "Minimality", enabling the systematic decomposition of the original complex problem into logically complete minimal subproblems while preserving key problem features and reducing reasoning length. We tested SoT across diverse datasets (e.g., GSM8K, MATH) and models (e.g., GPT-4o-mini, Qwen2.5), achieving inference accuracy that matches or surpasses mainstream CoTs standards. Additionally, by aligning the sampling process with algebraic constraints, our approach enhances the scalability of inference time in LLMs, ensuring both transparent reasoning and high performance. Our code will be publicly available at https://github.com/dlMARiA/Syzygy-of-thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11765v1">Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) face increasing inference latency as input context length and model size continue to grow. In particular, the retrieval-augmented generation (RAG) technique, which enhances LLM responses by incorporating external knowledge, exacerbates this issue by significantly increasing the number of input tokens. This expansion in token length leads to a substantial rise in computational overhead, particularly during the prefill stage, resulting in prolonged time-to-first-token (TTFT). To address this issue, this paper proposes a method to reduce TTFT by leveraging a disk-based key-value (KV) cache to lessen the computational burden during the prefill stage. We also introduce a disk-based shared KV cache management system, called Shared RAG-DCache, for multi-instance LLM RAG service environments. This system, together with an optimal system configuration, improves both throughput and latency under given resource constraints. Shared RAG-DCache exploits the locality of documents related to user queries in RAG, as well as the queueing delay in LLM inference services. It proactively generates and stores disk KV caches for query-related documents and shares them across multiple LLM instances to enhance inference performance. In experiments on a single host equipped with 2 GPUs and 1 CPU, Shared RAG-DCache achieved a 15~71% increase in throughput and up to a 12~65% reduction in latency, depending on the resource configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11750v1">Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted for ISPASS 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based inference workloads increasingly dominate data center costs and resource utilization. Therefore, understanding the inference workload characteristics on evolving CPU-GPU coupled architectures is crucial for optimization. This paper presents an in-depth analysis of LLM inference behavior on loosely-coupled (PCIe A100/H100) and closely-coupled (GH200) systems. We analyze performance dynamics using fine-grained operator-to-kernel trace analysis, facilitated by our novel profiler SKIP and metrics like Total Kernel Launch and Queuing Time (TKLQT). Results show that closely-coupled (CC) GH200 significantly outperforms loosely-coupled (LC) systems at large batch sizes, achieving 1.9x-2.7x faster prefill latency for Llama 3.2-1B. However, our analysis also reveals that GH200 remains CPU-bound up to 4x larger batch sizes than LC systems. In this extended CPU-bound region, we identify the performance characteristics of the Grace CPU as a key factor contributing to higher inference latency at low batch sizes on GH200. We demonstrate that TKLQT accurately identifies this CPU/GPU-bound transition point. Based on this analysis, we further show that kernel fusion offers significant potential to mitigate GH200's low-batch latency bottleneck by reducing kernel launch overhead. This detailed kernel-level characterization provides critical insights for optimizing diverse CPU-GPU coupling strategies. This work is an initial effort, and we plan to explore other major AI/DL workloads that demand different degrees of CPU-GPU heterogeneous architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11711v1">The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Static analysis is a cornerstone for software vulnerability detection, yet it often struggles with the classic precision-scalability trade-off. In practice, such tools often produce high false positive rates, particularly in large codebases like the Linux kernel. This imprecision can arise from simplified vulnerability modeling and over-approximation of path and data constraints. While large language models (LLMs) show promise in code understanding, their naive application to program analysis yields unreliable results due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly improves static analysis precision. BugLens guides an LLM to follow traditional analysis steps by assessing buggy code patterns for security impact and validating the constraints associated with static warnings. Evaluated on real-world Linux kernel bugs, BugLens raises precision from 0.10 (raw) and 0.50 (semi-automated refinement) to 0.72, substantially reducing false positives and revealing four previously unreported vulnerabilities. Our results suggest that a structured LLM-based workflow can meaningfully enhance the effectiveness of static analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11704v1">A Library of LLM Intrinsics for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11703v1">Progent: Programmable Privilege Control for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility. We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v2">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11671v1">Steering Prosocial AI Agents: Computational Basis of LLM's Decision Making in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12523v1">Memorization vs. Reasoning: Updating LLMs with New Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 9 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode vast amounts of pre-trained knowledge in their parameters, but updating them as real-world information evolves remains a challenge. Existing methodologies and benchmarks primarily target entity substitutions, failing to capture the full breadth of complex real-world dynamics. In this paper, we introduce Knowledge Update Playground (KUP), an automatic pipeline for simulating realistic knowledge updates reflected in an evidence corpora. KUP's evaluation framework includes direct and indirect probes to both test memorization of updated facts and reasoning over them, for any update learning methods. Next, we present a lightweight method called memory conditioned training (MCT), which conditions tokens in the update corpus on self-generated "memory" tokens during training. Our strategy encourages LLMs to surface and reason over newly memorized knowledge at inference. Our results on two strong LLMs show that (1) KUP benchmark is highly challenging, with the best CPT models achieving $<2\%$ in indirect probing setting (reasoning) and (2) MCT training significantly outperforms prior continued pre-training (CPT) baselines, improving direct probing (memorization) results by up to $25.4\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12522v1">Evaluating the Diversity and Quality of LLM Generated Content</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 ICLR 2025 Third Workshop on Deep Learning for Code
    </div>
    <details class="paper-abstract">
      Recent work suggests that preference-tuning techniques--including Reinforcement Learning from Human Preferences (RLHF) methods like PPO and GRPO, as well as alternatives like DPO--reduce diversity, creating a dilemma given that such models are widely deployed in applications requiring diverse outputs. To address this, we introduce a framework for measuring effective semantic diversity--diversity among outputs that meet quality thresholds--which better reflects the practical utility of large language models (LLMs). Using open-ended tasks that require no human intervention, we find counterintuitive results: although preference-tuned models--especially those trained via RL--exhibit reduced lexical and syntactic diversity, they produce greater effective semantic diversity than SFT or base models, not from increasing diversity among high-quality outputs, but from generating more high-quality outputs overall. We discover that preference tuning reduces syntactic diversity while preserving semantic diversity--revealing a distinction between diversity in form and diversity in content that traditional metrics often overlook. Our analysis further shows that smaller models are consistently more parameter-efficient at generating unique content within a fixed sampling budget, offering insights into the relationship between model scaling and diversity. These findings have important implications for applications that require diverse yet high-quality outputs, from creative assistance to synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04682v2">ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) advancements sparked a growing research interest in tool assisted LLMs solving real-world challenges, which calls for comprehensive evaluation of tool-use capabilities. While previous works focused on either evaluating over stateless web services (RESTful API), based on a single turn user prompt, or an off-policy dialog trajectory, ToolSandbox includes stateful tool execution, implicit state dependencies between tools, a built-in user simulator supporting on-policy conversational evaluation and a dynamic evaluation strategy for intermediate and final milestones over an arbitrary trajectory. We show that open source and proprietary models have a significant performance gap, and complex tasks like State Dependency, Canonicalization and Insufficient Information defined in ToolSandbox are challenging even the most capable SOTA LLMs, providing brand-new insights into tool-use LLM capabilities. ToolSandbox evaluation framework is released at https://github.com/apple/ToolSandbox
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09586v2">ValueCompass: A Framework for Measuring Contextual Value Alignment Between Human and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      As AI systems become more advanced, ensuring their alignment with a diverse range of individuals and societal values becomes increasingly critical. But how can we capture fundamental human values and assess the degree to which AI systems align with them? We introduce ValueCompass, a framework of fundamental values, grounded in psychological theory and a systematic review, to identify and evaluate human-AI alignment. We apply ValueCompass to measure the value alignment of humans and large language models (LLMs) across four real-world scenarios: collaborative writing, education, public sectors, and healthcare. Our findings reveal concerning misalignments between humans and LLMs, such as humans frequently endorse values like "National Security" which were largely rejected by LLMs. We also observe that values differ across scenarios, highlighting the need for context-aware AI alignment strategies. This work provides valuable insights into the design space of human-AI alignment, laying the foundations for developing AI systems that responsibly reflect societal values and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12491v1">Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01744v2">ALCM: Autonomous LLM-Augmented Causal Discovery Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      To perform effective causal inference in high-dimensional datasets, initiating the process with causal discovery is imperative, wherein a causal graph is generated based on observational data. However, obtaining a complete and accurate causal graph poses a formidable challenge, recognized as an NP- hard problem. Recently, the advent of Large Language Models (LLMs) has ushered in a new era, indicating their emergent capabilities and widespread applicability in facilitating causal reasoning across diverse domains, such as medicine, finance, and science. The expansive knowledge base of LLMs holds the potential to elevate the field of causal reasoning by offering interpretability, making inferences, generalizability, and uncovering novel causal structures. In this paper, we introduce a new framework, named Autonomous LLM-Augmented Causal Discovery Framework (ALCM), to synergize data-driven causal discovery algorithms and LLMs, automating the generation of a more resilient, accurate, and explicable causal graph. The ALCM consists of three integral components: causal structure learning, causal wrapper, and LLM-driven causal refiner. These components autonomously collaborate within a dynamic environment to address causal discovery questions and deliver plausible causal graphs. We evaluate the ALCM framework by implementing two demonstrations on seven well-known datasets. Experimental results demonstrate that ALCM outperforms existing LLM methods and conventional data-driven causal reasoning mechanisms. This study not only shows the effectiveness of the ALCM but also underscores new research directions in leveraging the causal reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12452v1">PlanGlow: Personalized Study Planning with an Explainable and Controllable LLM-Driven System</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 12 pages, 6 figures. To appear at ACM Learning@Scale 2025
    </div>
    <details class="paper-abstract">
      Personal development through self-directed learning is essential in today's fast-changing world, but many learners struggle to manage it effectively. While AI tools like large language models (LLMs) have the potential for personalized learning planning, they face issues such as transparency and hallucinated information. To address this, we propose PlanGlow, an LLM-based system that generates personalized, well-structured study plans with clear explanations and controllability through user-centered interactions. Through mixed methods, we surveyed 28 participants and interviewed 10 before development, followed by a within-subject experiment with 24 participants to evaluate PlanGlow's performance, usability, controllability, and explainability against two baseline systems: a GPT-4o-based system and Khan Academy's Khanmigo. Results demonstrate that PlanGlow significantly improves usability, explainability, and controllability. Additionally, two educational experts assessed and confirmed the quality of the generated study plans. These findings highlight PlanGlow's potential to enhance personalized learning and address key challenges in self-directed learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07466v2">IdentifyMe: A Challenging Long-Context Mention Resolution Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent evaluations of LLMs on coreference resolution have revealed that traditional output formats and evaluation metrics do not fully capture the models' referential understanding. To address this, we introduce IdentifyMe, a new benchmark for mention resolution presented in a multiple-choice question (MCQ) format, commonly used for evaluating LLMs. IdentifyMe features long narratives and employs heuristics to exclude easily identifiable mentions, creating a more challenging task. The benchmark also consists of a curated mixture of different mention types and corresponding entities, allowing for a fine-grained analysis of model performance. We evaluate both closed- and open source LLMs on IdentifyMe and observe a significant performance gap (20-30%) between the state-of-the-art sub-10B open models vs. closed ones. We observe that pronominal mentions, which have limited surface information, are typically much harder for models to resolve than nominal mentions. Additionally, we find that LLMs often confuse entities when their mentions overlap in nested structures. The highest-scoring model, GPT-4o, achieves 81.9% accuracy, highlighting the strong referential capabilities of state-of-the-art LLMs while also indicating room for further improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v6">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Oral, Multi-Agent AI in the Real World @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. TradingAgents is available at https://github.com/TauricResearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11511v1">Position Paper: Rethinking Privacy in RL for Sequential Decision-making in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Accepted to IJCNN 2025 Position Paper Track
    </div>
    <details class="paper-abstract">
      The rise of reinforcement learning (RL) in critical real-world applications demands a fundamental rethinking of privacy in AI systems. Traditional privacy frameworks, designed to protect isolated data points, fall short for sequential decision-making systems where sensitive information emerges from temporal patterns, behavioral strategies, and collaborative dynamics. Modern RL paradigms, such as federated RL (FedRL) and RL with human feedback (RLHF) in large language models (LLMs), exacerbate these challenges by introducing complex, interactive, and context-dependent learning environments that traditional methods do not address. In this position paper, we argue for a new privacy paradigm built on four core principles: multi-scale protection, behavioral pattern protection, collaborative privacy preservation, and context-aware adaptation. These principles expose inherent tensions between privacy, utility, and interpretability that must be navigated as RL systems become more pervasive in high-stakes domains like healthcare, autonomous vehicles, and decision support systems powered by LLMs. To tackle these challenges, we call for the development of new theoretical frameworks, practical mechanisms, and rigorous evaluation methodologies that collectively enable effective privacy protection in sequential decision-making systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12339v1">GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via A Dual-Branch Architecture</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-k layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11343v1">A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a prevailing approach for fine-tuning large language models (LLMs) on complex reasoning tasks. Among recent methods, GRPO stands out for its empirical success in training models such as DeepSeek-R1, yet the sources of its effectiveness remain poorly understood. In this work, we revisit GRPO from a reinforce-like algorithm perspective and analyze its core components. Surprisingly, we find that a simple rejection sampling baseline, RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO. Our ablation studies reveal that GRPO's main advantage arises from discarding prompts with entirely incorrect responses, rather than from its reward normalization. Motivated by this insight, we propose Reinforce-Rej, a minimal extension of policy gradient that filters both entirely incorrect and entirely correct samples. Reinforce-Rej improves KL efficiency and stability, serving as a lightweight yet effective alternative to more complex RL algorithms. We advocate RAFT as a robust and interpretable baseline, and suggest that future advances should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately. Our findings provide guidance for future work in reward-based LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11320v1">Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 42 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are indispensable in today's applications, but their inference procedure -- generating responses by processing text in segments and using a memory-heavy Key-Value (KV) cache -- demands significant computational resources, particularly under memory constraints. This paper formulates LLM inference optimization as a multi-stage online scheduling problem where sequential prompt arrivals and KV cache growth render conventional scheduling ineffective. We develop a fluid dynamics approximation to provide a tractable benchmark that guides algorithm design. Building on this, we propose the Waiting for Accumulated Inference Threshold (WAIT) algorithm, which uses multiple thresholds to schedule incoming prompts optimally when output lengths are known, and extend it to Nested WAIT for cases with unknown output lengths. Theoretical analysis shows that both algorithms achieve near-optimal performance against the fluid benchmark in heavy traffic conditions, balancing throughput, latency, and Time to First Token (TTFT). Experiments with the Llama-7B model on an A100 GPU using both synthetic and real-world datasets demonstrate improved throughput and reduced latency relative to established baselines like vLLM and Sarathi. This work bridges operations research and machine learning, offering a rigorous framework for the efficient deployment of LLMs under memory constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11281v1">The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11277v1">From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit excellent performance in natural language processing (NLP), but remain highly sensitive to the quality of input queries, especially when these queries contain misleading or inaccurate information. Existing methods focus on correcting the output, but they often overlook the potential of improving the ability of LLMs to detect and correct misleading content in the input itself. In this paper, we propose a novel three-stage fine-tuning method that enhances the ability of LLMs to detect and correct misleading information in the input, further improving response accuracy and reducing hallucinations. Specifically, the three stages include (1) training LLMs to identify misleading information, (2) training LLMs to correct the misleading information using built-in or external knowledge, and (3) training LLMs to generate accurate answers based on the corrected queries. To evaluate our method, we conducted experiments on three datasets for the hallucination detection task and the question answering (QA) task, as well as two datasets containing misleading information that we constructed. The experimental results demonstrate that our method significantly improves the accuracy and factuality of LLM responses, while also enhancing the ability to detect hallucinations and reducing the generation of hallucinations in the output, particularly when the query contains misleading information. We will publicly release our code upon acceptance.
    </details>
</div>
