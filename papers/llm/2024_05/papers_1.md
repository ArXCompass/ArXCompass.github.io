# llm - 2024_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00115v1">Towards LLM-Powered Verilog RTL Assistant: Self-Verification and Self-Correction</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      We explore the use of Large Language Models (LLMs) to generate high-quality Register-Transfer Level (RTL) code with minimal human interference. The traditional RTL design workflow requires human experts to manually write high-quality RTL code, which is time-consuming and error-prone. With the help of emerging LLMs, developers can describe their requirements to LLMs which then generate corresponding code in Python, C, Java, and more. Adopting LLMs to generate RTL design in hardware description languages is not trivial, given the complex nature of hardware design and the generated design has to meet the timing and physical constraints. We propose VeriAssist, an LLM-powered programming assistant for Verilog RTL design workflow. VeriAssist takes RTL design descriptions as input and generates high-quality RTL code with corresponding test benches. VeriAssist enables the LLM to self-correct and self-verify the generated code by adopting an automatic prompting system and integrating RTL simulator in the code generation loop. To generate an RTL design, VeriAssist first generates the initial RTL code and corresponding test benches, followed by a self-verification step that walks through the code with test cases to reason the code behavior at different time steps, and finally it self-corrects the code by reading the compilation and simulation results and generating final RTL code that fixes errors in compilation and simulation. This design fully leverages the LLMs' capabilities on multi-turn interaction and chain-of-thought reasoning to improve the quality of the generated code. We evaluate VeriAssist with various benchmark suites and find it significantly improves both syntax and functionality correctness over existing LLM implementations, thus minimizing human intervention and making RTL design more accessible to novice designers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00092v1">How Random is Random? Evaluating the Randomness and Humaness of LLMs' Coin Flips</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      One uniquely human trait is our inability to be random. We see and produce patterns where there should not be any and we do so in a predictable way. LLMs are supplied with human data and prone to human biases. In this work, we explore how LLMs approach randomness and where and how they fail through the lens of the well studied phenomena of generating binary random sequences. We find that GPT 4 and Llama 3 exhibit and exacerbate nearly every human bias we test in this context, but GPT 3.5 exhibits more random behavior. This dichotomy of randomness or humaness is proposed as a fundamental question of LLMs and that either behavior may be useful in different circumstances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.21030v1">Standards for Belief Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to demonstrate remarkable abilities across various domains, computer scientists are developing methods to understand their cognitive processes, particularly concerning how (and if) LLMs internally represent their beliefs about the world. However, this field currently lacks a unified theoretical foundation to underpin the study of belief in LLMs. This article begins filling this gap by proposing adequacy conditions for a representation in an LLM to count as belief-like. We argue that, while the project of belief measurement in LLMs shares striking features with belief measurement as carried out in decision theory and formal epistemology, it also differs in ways that should change how we measure belief. Thus, drawing from insights in philosophy and contemporary practices of machine learning, we establish four criteria that balance theoretical considerations with practical constraints. Our proposed criteria include accuracy, coherence, uniformity, and use, which together help lay the groundwork for a comprehensive understanding of belief representation in LLMs. We draw on empirical work showing the limitations of using various criteria in isolation to identify belief representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09085v5">The Earth is Flat because...: Investigating LLMs' Belief towards Misinformation via Persuasive Conversation</a></div>
    <div class="paper-meta">
      📅 2024-05-31
      | 💬 Accepted to ACL'24 (Main). Camera-ready version
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encapsulate vast amounts of knowledge but still remain vulnerable to external misinformation. Existing research mainly studied this susceptibility behavior in a single-turn setting. However, belief can change during a multi-turn conversation, especially a persuasive one. Therefore, in this study, we delve into LLMs' susceptibility to persuasive conversations, particularly on factual questions that they can answer correctly. We first curate the Farm (i.e., Fact to Misinform) dataset, which contains factual questions paired with systematically generated persuasive misinformation. Then, we develop a testing framework to track LLMs' belief changes in a persuasive dialogue. Through extensive experiments, we find that LLMs' correct beliefs on factual knowledge can be easily manipulated by various persuasive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20859v1">clembench-2024: A Challenging, Dynamic, Complementary, Multilingual Benchmark and Underlying Flexible Framework for LLMs as Multi-Action Agents</a></div>
    <div class="paper-meta">
      📅 2024-05-31
      | 💬 under review
    </div>
    <details class="paper-abstract">
      It has been established in recent work that Large Language Models (LLMs) can be prompted to "self-play" conversational games that probe certain capabilities (general instruction following, strategic goal orientation, language understanding abilities), where the resulting interactive game play can be automatically scored. In this paper, we take one of the proposed frameworks for setting up such game-play environments, and further test its usefulness as an evaluation instrument, along a number of dimensions: We show that it can easily keep up with new developments while avoiding data contamination, we show that the tests implemented within it are not yet saturated (human performance is substantially higher than that of even the best models), and we show that it lends itself to investigating additional questions, such as the impact of the prompting language on performance. We believe that the approach forms a good basis for making decisions on model choice for building applied interactive systems, and perhaps ultimately setting up a closed-loop development environment of system and simulated evaluator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18870v2">LLMs achieve adult human performance on higher-order theory of mind tasks</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      This paper examines the extent to which large language models (LLMs) have developed higher-order theory of mind (ToM); the human ability to reason about multiple mental and emotional states in a recursive manner (e.g. I think that you believe that she knows). This paper builds on prior work by introducing a handwritten test suite -- Multi-Order Theory of Mind Q&A -- and using it to compare the performance of five LLMs to a newly gathered adult human benchmark. We find that GPT-4 and Flan-PaLM reach adult-level and near adult-level performance on ToM tasks overall, and that GPT-4 exceeds adult performance on 6th order inferences. Our results suggest that there is an interplay between model size and finetuning for the realisation of ToM abilities, and that the best-performing LLMs have developed a generalised capacity for ToM. Given the role that higher-order ToM plays in a wide range of cooperative and competitive human behaviours, these findings have significant implications for user-facing LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20701v1">Unveiling the Lexical Sensitivity of LLMs: Combinatorial Optimization for Prompt Enhancement</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate exceptional instruct-following ability to complete various downstream tasks. Although this impressive ability makes LLMs flexible task solvers, their performance in solving tasks also heavily relies on instructions. In this paper, we reveal that LLMs are over-sensitive to lexical variations in task instructions, even when the variations are imperceptible to humans. By providing models with neighborhood instructions, which are closely situated in the latent representation space and differ by only one semantically similar word, the performance on downstream tasks can be vastly different. Following this property, we propose a black-box Combinatorial Optimization framework for Prompt Lexical Enhancement (COPLE). COPLE performs iterative lexical optimization according to the feedback from a batch of proxy tasks, using a search strategy related to word influence. Experiments show that even widely-used human-crafted prompts for current benchmarks suffer from the lexical sensitivity of models, and COPLE recovers the declined model ability in both instruct-following and solving downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20681v1">No Free Lunch Theorem for Privacy-Preserving LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      Individuals and businesses have been significantly benefited by Large Language Models (LLMs) including PaLM, Gemini and ChatGPT in various ways. For example, LLMs enhance productivity, reduce costs, and enable us to focus on more valuable tasks. Furthermore, LLMs possess the capacity to sift through extensive datasets, uncover underlying patterns, and furnish critical insights that propel the frontiers of technology and science. However, LLMs also pose privacy concerns. Users' interactions with LLMs may expose their sensitive personal or company information. A lack of robust privacy safeguards and legal frameworks could permit the unwarranted intrusion or improper handling of individual data, thereby risking infringements of privacy and the theft of personal identities. To ensure privacy, it is essential to minimize the dependency between shared prompts and private information. Various randomization approaches have been proposed to protect prompts' privacy, but they may incur utility loss compared to unprotected LLMs prompting. Therefore, it is essential to evaluate the balance between the risk of privacy leakage and loss of utility when conducting effective protection mechanisms. The current study develops a framework for inferring privacy-protected Large Language Models (LLMs) and lays down a solid theoretical basis for examining the interplay between privacy preservation and utility. The core insight is encapsulated within a theorem that is called as the NFL (abbreviation of the word No-Free-Lunch) Theorem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18339v2">When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications</a></div>
    <div class="paper-meta">
      📅 2024-05-31
      | 💬 accepted by SIGIR'24
    </div>
    <details class="paper-abstract">
      The recent surge in Large Language Models (LLMs) has garnered significant attention across numerous fields. Fine-tuning is often required to fit general LLMs for a specific domain, like the web-based healthcare system. However, two problems arise during fine-tuning LLMs for medical applications. One is the task variety problem, which involves distinct tasks in real-world medical scenarios. The variety often leads to sub-optimal fine-tuning for data imbalance and seesaw problems. Besides, the large amount of parameters in LLMs leads to huge time and computation consumption by fine-tuning. To address these two problems, we propose a novel parameter efficient fine-tuning framework for multi-task medical applications, dubbed as MOELoRA. The designed framework aims to absorb both the benefits of mixture-of-expert (MOE) for multi-task learning and low-rank adaptation (LoRA) for parameter efficient fine-tuning. For unifying MOE and LoRA, we devise multiple experts as the trainable parameters, where each expert consists of a pair of low-rank matrices to retain the small size of trainable parameters. Then, a task-motivated gate function for all MOELoRA layers is proposed, which can control the contributions of each expert and produce distinct parameters for various tasks. We conduct experiments on a multi-task medical dataset, indicating MOELoRA outperforms the existing parameter efficient fine-tuning methods. The code is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20625v1">Robust Planning with LLM-Modulo Framework: Case Study in Travel Planning</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      As the applicability of Large Language Models (LLMs) extends beyond traditional text processing tasks, there is a burgeoning interest in their potential to excel in planning and reasoning assignments, realms traditionally reserved for System 2 cognitive competencies. Despite their perceived versatility, the research community is still unraveling effective strategies to harness these models in such complex domains. The recent discourse introduced by the paper on LLM Modulo marks a significant stride, proposing a conceptual framework that enhances the integration of LLMs into diverse planning and reasoning activities. This workshop paper delves into the practical application of this framework within the domain of travel planning, presenting a specific instance of its implementation. We are using the Travel Planning benchmark by the OSU NLP group, a benchmark for evaluating the performance of LLMs in producing valid itineraries based on user queries presented in natural language. While popular methods of enhancing the reasoning abilities of LLMs such as Chain of Thought, ReAct, and Reflexion achieve a meager 0%, 0.6%, and 0% with GPT3.5-Turbo respectively, our operationalization of the LLM-Modulo framework for TravelPlanning domain provides a remarkable improvement, enhancing baseline performances by 4.6x for GPT4-Turbo and even more for older models like GPT3.5-Turbo from 0% to 5%. Furthermore, we highlight the other useful roles of LLMs in the planning pipeline, as suggested in LLM-Modulo, which can be reliably operationalized such as extraction of useful critics and reformulator for critics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12146v3">Enabling Weak LLMs to Judge Response Reliability via Meta Ranking</a></div>
    <div class="paper-meta">
      📅 2024-05-31
      | 💬 Preprint, under review. 28 pages
    </div>
    <details class="paper-abstract">
      Despite the strong performance of large language models (LLMs) across a wide range of tasks, they still have reliability issues. Previous studies indicate that strong LLMs like GPT-4-turbo excel in evaluating the reliability of responses from LLMs, but face efficiency and local deployment issues. Thus, to enable weak LLMs to effectively assess the reliability of LLM responses, we propose a novel cross-query-comparison-based method called $\textit{Meta Ranking}$ (MR). Unlike previous few-shot methods that solely based on in-context learning capabilities in LLMs, MR assesses reliability by pairwisely ranking the target query-response pair with multiple reference query-response pairs. We found that MR is highly effective in error detection for LLM responses, where weak LLMs, such as Phi-2, could surpass strong baselines like GPT-3.5-turbo, requiring only five reference samples and significantly improving efficiency. We further demonstrate that MR can enhance strong LLMs' performance in two practical applications: model cascading and instruction tuning. In model cascading, we combine open- and closed-source LLMs to achieve performance comparable to GPT-4-turbo with lower costs. In instruction tuning, we use MR for iterative training data filtering, significantly reducing data processing time and enabling LLaMA-7B and Phi-2 to surpass Alpaca-13B with fewer training tokens. These results underscore the high potential of MR in both efficiency and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19325v2">Nearest Neighbor Speculative Decoding for LLM Generation and Attribution</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often hallucinate and lack the ability to provide attribution for their generations. Semi-parametric LMs, such as kNN-LM, approach these limitations by refining the output of an LM for a given prompt using its nearest neighbor matches in a non-parametric data store. However, these models often exhibit slow inference speeds and produce non-fluent texts. In this paper, we introduce Nearest Neighbor Speculative Decoding (NEST), a novel semi-parametric language modeling approach that is capable of incorporating real-world text spans of arbitrary length into the LM generations and providing attribution to their sources. NEST performs token-level retrieval at each inference step to compute a semi-parametric mixture distribution and identify promising span continuations in a corpus. It then uses an approximate speculative decoding procedure that accepts a prefix of the retrieved span or generates a new token. NEST significantly enhances the generation quality and attribution rate of the base LM across a variety of knowledge-intensive tasks, surpassing the conventional kNN-LM method and performing competitively with in-context retrieval augmentation. In addition, NEST substantially improves the generation speed, achieving a 1.8x speedup in inference time when applied to Llama-2-Chat 70B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.15255v4">Spoken Question Answering and Speech Continuation Using Spectrogram-Powered LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-31
      | 💬 ICLR 2024 camera-ready
    </div>
    <details class="paper-abstract">
      We present Spectron, a novel approach to adapting pre-trained large language models (LLMs) to perform spoken question answering (QA) and speech continuation. By endowing the LLM with a pre-trained speech encoder, our model becomes able to take speech inputs and generate speech outputs. The entire system is trained end-to-end and operates directly on spectrograms, simplifying our architecture. Key to our approach is a training objective that jointly supervises speech recognition, text continuation, and speech synthesis using only paired speech-text pairs, enabling a `cross-modal' chain-of-thought within a single decoding pass. Our method surpasses existing spoken language models in speaker preservation and semantic coherence. Furthermore, the proposed model improves upon direct initialization in retaining the knowledge of the original LLM as demonstrated through spoken QA datasets. We release our audio samples (https://michelleramanovich.github.io/spectron/spectron) and spoken QA dataset (https://github.com/google-research-datasets/LLAMA1-Test-Set).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20551v1">EM-Assist: Safe Automated ExtractMethod Refactoring with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-31
      | 💬 This paper is accepted to the tool demonstration track of the 32nd ACM Symposium on the Foundations of Software Engineering (FSE 2024). This is an author copy
    </div>
    <details class="paper-abstract">
      Excessively long methods, loaded with multiple responsibilities, are challenging to understand, debug, reuse, and maintain. The solution lies in the widely recognized Extract Method refactoring. While the application of this refactoring is supported in modern IDEs, recommending which code fragments to extract has been the topic of many research tools. However, they often struggle to replicate real-world developer practices, resulting in recommendations that do not align with what a human developer would do in real life. To address this issue, we introduce EM-Assist, an IntelliJ IDEA plugin that uses LLMs to generate refactoring suggestions and subsequently validates, enhances, and ranks them. Finally, EM-Assist uses the IntelliJ IDE to apply the user-selected recommendation. In our extensive evaluation of 1,752 real-world refactorings that actually took place in open-source projects, EM-Assist's recall rate was 53.4% among its top-5 recommendations, compared to 39.4% for the previous best-in-class tool that relies solely on static analysis. Moreover, we conducted a usability survey with 18 industrial developers and 94.4% gave a positive rating.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12459v2">Towards Socially and Morally Aware RL agent: Reward Design With LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      When we design and deploy an Reinforcement Learning (RL) agent, reward functions motivates agents to achieve an objective. An incorrect or incomplete specification of the objective can result in behavior that does not align with human values - failing to adhere with social and moral norms that are ambiguous and context dependent, and cause undesired outcomes such as negative side effects and exploration that is unsafe. Previous work have manually defined reward functions to avoid negative side effects, use human oversight for safe exploration, or use foundation models as planning tools. This work studies the ability of leveraging Large Language Models (LLM)' understanding of morality and social norms on safe exploration augmented RL methods. This work evaluates language model's result against human feedbacks and demonstrates language model's capability as direct reward signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20450v1">Decentralized AI: Permissionless LLM Inference on POKT Network</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      POKT Network's decentralized Remote Procedure Call (RPC) infrastructure, surpassing 740 billion requests since launching on MainNet in 2020, is well-positioned to extend into providing AI inference services with minimal design or implementation modifications. This litepaper illustrates how the network's open-source and permissionless design aligns incentives among model researchers, hardware operators, API providers and users whom we term model Sources, Suppliers, Gateways and Applications respectively. Through its Relay Mining algorithm, POKT creates a transparent marketplace where costs and earnings directly reflect cryptographically verified usage. This decentralized framework offers large model AI researchers a new avenue to disseminate their work and generate revenue without the complexities of maintaining infrastructure or building end-user products. Supply scales naturally with demand, as evidenced in recent years and the protocol's free market dynamics. POKT Gateways facilitate network growth, evolution, adoption, and quality by acting as application-facing load balancers, providing value-added features without managing LLM nodes directly. This vertically decoupled network, battle tested over several years, is set up to accelerate the adoption, operation, innovation and financialization of open-source models. It is the first mature permissionless network whose quality of service competes with centralized entities set up to provide application grade inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20434v1">Facilitating Human-LLM Collaboration through Factuality Scores and Source Attributions</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Submitted to the Trust and Reliance in Evolving Human-AI Workflows (TREW) Workshop at CHI 2024
    </div>
    <details class="paper-abstract">
      While humans increasingly rely on large language models (LLMs), they are susceptible to generating inaccurate or false information, also known as "hallucinations". Technical advancements have been made in algorithms that detect hallucinated content by assessing the factuality of the model's responses and attributing sections of those responses to specific source documents. However, there is limited research on how to effectively communicate this information to users in ways that will help them appropriately calibrate their trust toward LLMs. To address this issue, we conducted a scenario-based study (N=104) to systematically compare the impact of various design strategies for communicating factuality and source attribution on participants' ratings of trust, preferences, and ease in validating response accuracy. Our findings reveal that participants preferred a design in which phrases within a response were color-coded based on the computed factuality scores. Additionally, participants increased their trust ratings when relevant sections of the source material were highlighted or responses were annotated with reference numbers corresponding to those sources, compared to when they received no annotation in the source material. Our study offers practical design guidelines to facilitate human-LLM collaboration and it promotes a new human role to carefully evaluate and take responsibility for their use of LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20335v1">Xwin-LM: Strong and Scalable Alignment Practice for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      In this work, we present Xwin-LM, a comprehensive suite of alignment methodologies for large language models (LLMs). This suite encompasses several key techniques, including supervised finetuning (SFT), reward modeling (RM), rejection sampling finetuning (RS), and direct preference optimization (DPO). The key components are as follows: (1) Xwin-LM-SFT, models initially finetuned with high-quality instruction data; (2) Xwin-Pair, a large-scale, multi-turn preference dataset meticulously annotated using GPT-4; (3) Xwin-RM, reward models trained on Xwin-Pair, developed at scales of 7B, 13B, and 70B parameters; (4) Xwin-Set, a multiwise preference dataset in which each prompt is linked to 64 unique responses generated by Xwin-LM-SFT and scored by Xwin-RM; (5) Xwin-LM-RS, models finetuned with the highest-scoring responses from Xwin-Set; (6) Xwin-LM-DPO, models further optimized on Xwin-Set using the DPO algorithm. Our evaluations on AlpacaEval and MT-bench demonstrate consistent and significant improvements across the pipeline, demonstrating the strength and scalability of Xwin-LM. The repository https://github.com/Xwin-LM/Xwin-LM will be continually updated to foster community research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12125v3">CodeTailor: LLM-Powered Personalized Parsons Puzzles for Engaging Support While Learning Programming</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Accepted to the Eleventh ACM Conference on Learning @ Scale (L@S 24) as a full paper
    </div>
    <details class="paper-abstract">
      Learning to program can be challenging, and providing high-quality and timely support at scale is hard. Generative AI and its products, like ChatGPT, can create a solution for most intro-level programming problems. However, students might use these tools to just generate code for them, resulting in reduced engagement and limited learning. In this paper, we present CodeTailor, a system that leverages a large language model (LLM) to provide personalized help to students while still encouraging cognitive engagement. CodeTailor provides a personalized Parsons puzzle to support struggling students. In a Parsons puzzle, students place mixed-up code blocks in the correct order to solve a problem. A technical evaluation with previous incorrect student code snippets demonstrated that CodeTailor could deliver high-quality (correct, personalized, and concise) Parsons puzzles based on their incorrect code. We conducted a within-subjects study with 18 novice programmers. Participants perceived CodeTailor as more engaging than just receiving an LLM-generated solution (the baseline condition). In addition, participants applied more supported elements from the scaffolded practice to the posttest when using CodeTailor than baseline. Overall, most participants preferred using CodeTailor versus just receiving the LLM-generated code for learning. Qualitative observations and interviews also provided evidence for the benefits of CodeTailor, including thinking more about solution construction, fostering continuity in learning, promoting reflection, and boosting confidence. We suggest future design ideas to facilitate active learning opportunities with generative AI techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20202v1">One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced rapidly but face significant memory demands. While quantization has shown promise for LLMs, current methods typically require lengthy training to alleviate the performance degradation from quantization loss. However, deploying LLMs across diverse scenarios with different resource constraints, e.g., servers and personal computers, requires repeated training per application, which amplifies the lengthy training problem. Given that, it is advantageous to train a once-for-all (OFA) supernet capable of yielding diverse optimal subnets for downstream applications through one-shot training. Nonetheless, the scale of current language models impedes efficiency and amplifies interference from weight sharing between subnets. We make an initial attempt to extend the once-for-all framework to large language models. Specifically, we decouple shared weights to eliminate the interference and incorporate Low-Rank adapters for training efficiency. Furthermore, we observe the imbalance allocation of training resources from the traditional uniform sampling. A non-parametric scheduler is introduced to adjust the sampling rate for each quantization configuration, achieving a more balanced allocation among subnets with varying demands. We validate the approach on LLaMA2 families, and downstream evaluation confirms our ability to maintain high performance while significantly reducing deployment time faced with multiple scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20189v1">Nadine: An LLM-driven Intelligent Social Robot with Affective Capabilities and Human-like Memory</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      In this work, we describe our approach to developing an intelligent and robust social robotic system for the Nadine social robot platform. We achieve this by integrating Large Language Models (LLMs) and skilfully leveraging the powerful reasoning and instruction-following capabilities of these types of models to achieve advanced human-like affective and cognitive capabilities. This approach is novel compared to the current state-of-the-art LLM-based agents which do not implement human-like long-term memory or sophisticated emotional appraisal. The naturalness of social robots, consisting of multiple modules, highly depends on the performance and capabilities of each component of the system and the seamless integration of the components. We built a social robot system that enables generating appropriate behaviours through multimodal input processing, bringing episodic memories accordingly to the recognised user, and simulating the emotional states of the robot induced by the interaction with the human partner. In particular, we introduce an LLM-agent frame for social robots, SoR-ReAct, serving as a core component for the interaction module in our system. This design has brought forth the advancement of social robots and aims to increase the quality of human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20163v1">Reasoning about concepts with LLMs: Inconsistencies abound</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 15 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      The ability to summarize and organize knowledge into abstract concepts is key to learning and reasoning. Many industrial applications rely on the consistent and systematic use of concepts, especially when dealing with decision-critical knowledge. However, we demonstrate that, when methodically questioned, large language models (LLMs) often display and demonstrate significant inconsistencies in their knowledge. Computationally, the basic aspects of the conceptualization of a given domain can be represented as Is-A hierarchies in a knowledge graph (KG) or ontology, together with a few properties or axioms that enable straightforward reasoning. We show that even simple ontologies can be used to reveal conceptual inconsistencies across several LLMs. We also propose strategies that domain experts can use to evaluate and improve the coverage of key domain concepts in LLMs of various sizes. In particular, we have been able to significantly enhance the performance of LLMs of various sizes with openly available weights using simple knowledge-graph (KG) based prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14852v2">PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      There has been significant interest in "extreme" compression of large language models (LLMs), i.e., to 1-2 bits per parameter, which allows such models to be executed efficiently on resource-constrained devices. Existing work focused on improved one-shot quantization techniques and weight representations; yet, purely post-training approaches are reaching diminishing returns in terms of the accuracy-vs-bit-width trade-off. State-of-the-art quantization methods such as QuIP# and AQLM include fine-tuning (part of) the compressed parameters over a limited amount of calibration data; however, such fine-tuning techniques over compressed weights often make exclusive use of straight-through estimators (STE), whose performance is not well-understood in this setting. In this work, we question the use of STE for extreme LLM compression, showing that it can be sub-optimal, and perform a systematic study of quantization-aware fine-tuning strategies for LLMs. We propose PV-Tuning - a representation-agnostic framework that generalizes and improves upon existing fine-tuning strategies, and provides convergence guarantees in restricted cases. On the practical side, when used for 1-2 bit vector quantization, PV-Tuning outperforms prior techniques for highly-performant models such as Llama and Mistral. Using PV-Tuning, we achieve the first Pareto-optimal quantization for Llama 2 family models at 2 bits per parameter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20099v1">Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      Safety, security, and compliance are essential requirements when aligning large language models (LLMs). However, many seemingly aligned LLMs are soon shown to be susceptible to jailbreak attacks. These attacks aim to circumvent the models' safety guardrails and security mechanisms by introducing jailbreak prompts into malicious queries. In response to these challenges, this paper introduces Defensive Prompt Patch (DPP), a novel prompt-based defense mechanism specifically designed to protect LLMs against such sophisticated jailbreak strategies. Unlike previous approaches, which have often compromised the utility of the model for the sake of safety, DPP is designed to achieve a minimal Attack Success Rate (ASR) while preserving the high utility of LLMs. Our method uses strategically designed interpretable suffix prompts that effectively thwart a wide range of standard and adaptive jailbreak techniques. Empirical results conducted on LLAMA-2-7B-Chat and Mistral-7B-Instruct-v0.2 models demonstrate the robustness and adaptability of DPP, showing significant reductions in ASR with negligible impact on utility. Our approach not only outperforms existing defense strategies in balancing safety and functionality, but also provides a scalable and interpretable solution applicable to various LLM platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01567v2">ReMatch: Retrieval Enhanced Schema Matching with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      Schema matching is a crucial task in data integration, involving the alignment of a source schema with a target schema to establish correspondence between their elements. This task is challenging due to textual and semantic heterogeneity, as well as differences in schema sizes. Although machine-learning-based solutions have been explored in numerous studies, they often suffer from low accuracy, require manual mapping of the schemas for model training, or need access to source schema data which might be unavailable due to privacy concerns. In this paper we present a novel method, named ReMatch, for matching schemas using retrieval-enhanced Large Language Models (LLMs). Our method avoids the need for predefined mapping, any model training, or access to data in the source database. Our experimental results on large real-world schemas demonstrate that ReMatch is an effective matcher. By eliminating the requirement for training data, ReMatch becomes a viable solution for real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.12392v3">PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Our code and data are at https://github.com/Jiuzhouh/PiVe (accepted to ACL 2024 Findings)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great abilities of solving various natural language tasks in different domains. Due to the training objective of LLMs and their pre-training data, LLMs are not very well equipped for tasks involving structured data generation. We propose a framework, Prompting with Iterative Verification (PiVe), to improve graph-based generative capability of LLMs. We show how a small language model could be trained to act as a verifier module for the output of an LLM~(i.e., ChatGPT, GPT-4), and to iteratively improve its performance via fine-grained corrective instructions. We also show how the verifier module could apply iterative corrections offline for a more cost-effective solution to the text-to-graph generation task. Experiments on three graph-based datasets show consistent improvement gained via PiVe. Additionally, we create GenWiki-HIQ and highlight that the verifier module can be used as a data augmentation tool to help improve the quality of automatically generated parallel text-graph datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20787v1">PGA-SciRE: Harnessing LLM on Data Augmentation for Enhancing Scientific Relation Extraction</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      Relation Extraction (RE) aims at recognizing the relation between pairs of entities mentioned in a text. Advances in LLMs have had a tremendous impact on NLP. In this work, we propose a textual data augmentation framework called PGA for improving the performance of models for RE in the scientific domain. The framework introduces two ways of data augmentation, utilizing a LLM to obtain pseudo-samples with the same sentence meaning but with different representations and forms by paraphrasing the original training set samples. As well as instructing LLM to generate sentences that implicitly contain information about the corresponding labels based on the relation and entity of the original training set samples. These two kinds of pseudo-samples participate in the training of the RE model together with the original dataset, respectively. The PGA framework in the experiment improves the F1 scores of the three mainstream models for RE within the scientific domain. Also, using a LLM to obtain samples can effectively reduce the cost of manually labeling data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20015v1">Efficient LLM-Jailbreaking by Introducing Visual Modality</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreaks that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) through the incorporation of a visual module into the target LLM. Subsequently, we conduct an efficient MLLM-jailbreak to generate jailbreaking embeddings embJS. Finally, we convert the embJS into text space to facilitate the jailbreaking of the target LLM. Compared to direct LLM-jailbreaking, our approach is more efficient, as MLLMs are more vulnerable to jailbreaking than pure LLM. Additionally, to improve the attack success rate (ASR) of jailbreaking, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class jailbreaking capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20003v1">Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      Uncertainty quantification in Large Language Models (LLMs) is crucial for applications where safety and reliability are important. In particular, uncertainty can be used to improve the trustworthiness of LLMs by detecting factually incorrect model responses, commonly called hallucinations. Critically, one should seek to capture the model's semantic uncertainty, i.e., the uncertainty over the meanings of LLM outputs, rather than uncertainty over lexical or syntactic variations that do not affect answer correctness. To address this problem, we propose Kernel Language Entropy (KLE), a novel method for uncertainty estimation in white- and black-box LLMs. KLE defines positive semidefinite unit trace kernels to encode the semantic similarities of LLM outputs and quantifies uncertainty using the von Neumann entropy. It considers pairwise semantic dependencies between answers (or semantic clusters), providing more fine-grained uncertainty estimates than previous methods based on hard clustering of answers. We theoretically prove that KLE generalizes the previous state-of-the-art method called semantic entropy and empirically demonstrate that it improves uncertainty quantification performance across multiple natural language generation datasets and LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12052v3">Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Accepted by ACL 2024 main conference. Repo: https://github.com/plageon/SlimPLM
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) and search engines represents a significant evolution in knowledge acquisition methodologies. However, determining the knowledge that an LLM already possesses and the knowledge that requires the help of a search engine remains an unresolved issue. Most existing methods solve this problem through the results of preliminary answers or reasoning done by the LLM itself, but this incurs excessively high computational costs. This paper introduces a novel collaborative approach, namely SlimPLM, that detects missing knowledge in LLMs with a slim proxy model, to enhance the LLM's knowledge acquisition process. We employ a proxy model which has far fewer parameters, and take its answers as heuristic answers. Heuristic answers are then utilized to predict the knowledge required to answer the user question, as well as the known and unknown knowledge within the LLM. We only conduct retrieval for the missing knowledge in questions that the LLM does not know. Extensive experimental results on five datasets with two LLMs demonstrate a notable improvement in the end-to-end performance of LLMs in question-answering tasks, achieving or surpassing current state-of-the-art models with lower LLM inference costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12174v2">BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Accepted by ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Retrieval-augmented large language models (LLMs) have demonstrated efficacy in knowledge-intensive tasks such as open-domain QA, addressing inherent challenges in knowledge update and factual inadequacy. However, inconsistencies between retrieval knowledge and the necessary knowledge for LLMs, leading to a decline in LLM's answer quality. This paper introduces BIDER, an approach that refines retrieval documents into Key Supporting Evidence (KSE) through knowledge synthesis, supervised fine-tuning (SFT), and preference alignment. We train BIDER by learning from crafting KSE, while maximizing its output to align with LLM's information acquisition preferences through reinforcement learning. Evaluations across five datasets show BIDER boosts LLMs' answer quality by 7% while reducing input content length in retrieval documents by 80%, outperforming existing methods. The proposed KSE simulation effectively equips LLMs with essential information for accurate question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19954v1">GenKubeSec: LLM-Based Kubernetes Misconfiguration Detection, Localization, Reasoning, and Remediation</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      A key challenge associated with Kubernetes configuration files (KCFs) is that they are often highly complex and error-prone, leading to security vulnerabilities and operational setbacks. Rule-based (RB) tools for KCF misconfiguration detection rely on static rule sets, making them inherently limited and unable to detect newly-discovered misconfigurations. RB tools also suffer from misdetection, since mistakes are likely when coding the detection rules. Recent methods for detecting and remediating KCF misconfigurations are limited in terms of their scalability and detection coverage, or due to the fact that they have high expertise requirements and do not offer automated remediation along with misconfiguration detection. Novel approaches that employ LLMs in their pipeline rely on API-based, general-purpose, and mainly commercial models. Thus, they pose security challenges, have inconsistent classification performance, and can be costly. In this paper, we propose GenKubeSec, a comprehensive and adaptive, LLM-based method, which, in addition to detecting a wide variety of KCF misconfigurations, also identifies the exact location of the misconfigurations and provides detailed reasoning about them, along with suggested remediation. When empirically compared with three industry-standard RB tools, GenKubeSec achieved equivalent precision (0.990) and superior recall (0.999). When a random sample of KCFs was examined by a Kubernetes security expert, GenKubeSec's explanations as to misconfiguration localization, reasoning and remediation were 100% correct, informative and useful. To facilitate further advancements in this domain, we share the unique dataset we collected, a unified misconfiguration index we developed for label standardization, our experimentation code, and GenKubeSec itself as an open-source tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02446v3">LQER: Low-Rank Quantization Error Reconstruction for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Accepted at ICML2024
    </div>
    <details class="paper-abstract">
      Post-training quantization of Large Language Models (LLMs) is challenging. In this work, we introduce Low-rank Quantization Error Reduction (LQER), which combines quantization and low-rank approximation to recover the model capability. LQER leverages an activation-induced scale matrix to drive the singular value distribution of quantization error towards a desirable distribution, which enables nearly-lossless W4A8 quantization on various LLMs and downstream tasks without the need for knowledge distillation, grid search, or gradient-base iterative optimization. Unlike existing methods, the computation pattern of LQER eliminates the need for specialized Scatter and Gather processes to collect high-precision weights from irregular memory locations. Our W4A8 LLMs achieve near-lossless performance on six popular downstream tasks, while using 1.36$\times$ fewer hardware resources than the leading state-of-the-art method. We open-source our framework at https://github.com/ChengZhang-98/lqer
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19888v1">Parrot: Efficient Serving of LLM-based Applications with Semantic Variable</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 To appear on USENIX OSDI 2024
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has enabled LLM-based applications (a.k.a. AI agents or co-pilots), a new software paradigm that combines the strength of LLM and conventional software. Diverse LLM applications from different tenants could design complex workflows using multiple LLM requests to accomplish one task. However, they have to use the over-simplified request-level API provided by today's public LLM services, losing essential application-level information. Public LLM services have to blindly optimize individual LLM requests, leading to sub-optimal end-to-end performance of LLM applications. This paper introduces Parrot, an LLM service system that focuses on the end-to-end experience of LLM-based applications. Parrot proposes Semantic Variable, a unified abstraction to expose application-level knowledge to public LLM services. A Semantic Variable annotates an input/output variable in the prompt of a request, and creates the data pipeline when connecting multiple LLM requests, providing a natural way to program LLM applications. Exposing Semantic Variables to the public LLM service allows it to perform conventional data flow analysis to uncover the correlation across multiple LLM requests. This correlation opens a brand-new optimization space for the end-to-end performance of LLM-based applications. Extensive evaluations demonstrate that Parrot can achieve up to an order-of-magnitude improvement for popular and practical use cases of LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19635v1">GKT: A Novel Guidance-Based Knowledge Transfer Framework For Efficient Cloud-edge Collaboration LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      The burgeoning size of Large Language Models (LLMs) has led to enhanced capabilities in generating responses, albeit at the expense of increased inference times and elevated resource demands. Existing methods of acceleration, predominantly hinged on knowledge distillation, generally necessitate fine-tuning of considerably large models, such as Llama-7B, posing a challenge for average users. Furthermore, present techniques for expediting inference and reducing costs operate independently. To address these issues, we introduce a novel and intuitive Guidance-based Knowledge Transfer (GKT) framework. This approach leverages a larger LLM as a ''teacher'' to create guidance prompts, paired with a smaller ''student'' model to finalize responses. Remarkably, GKT requires no fine-tuning and doesn't necessitate the teacher and student models to have the same vocabulary, allowing for extensive batch generation to accelerate the process while ensuring user customization. GKT can be seamlessly integrated into cloud-edge collaboration architectures, and is versatile enough for plug-and-play application across various models. It excels in both efficiency and affordability, epitomizing a ''cheap and cheerful'' solution. GKT achieves a maximum accuracy improvement of 14.18%, along with a 10.72 times speed-up on GSM8K and an accuracy improvement of 14.00 % along with a 7.73 times speed-up in CSQA. When utilizing ChatGPT as teacher model and Llama2-70B as the student model, we can achieve 95.00% of ChatGPT's performance at 52% of the cost. The results highlight substantial enhancements in accuracy and processing speed on the GSM8K and CSQA datasets, surpassing the performance of using either the student or teacher models in isolation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13494v2">GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis</a></div>
    <div class="paper-meta">
      📅 2024-05-29
      | 💬 Accepted to ACL 2024 Main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for detecting jailbreak prompts are primarily online moderation APIs or finetuned LLMs. These strategies, however, often require extensive and resource-intensive data collection and training processes. In this study, we propose GradSafe, which effectively detects jailbreak prompts by scrutinizing the gradients of safety-critical parameters in LLMs. Our method is grounded in a pivotal observation: the gradients of an LLM's loss for jailbreak prompts paired with compliance response exhibit similar patterns on certain safety-critical parameters. In contrast, safe prompts lead to different gradient patterns. Building on this observation, GradSafe analyzes the gradients from prompts (paired with compliance responses) to accurately detect jailbreak prompts. We show that GradSafe, applied to Llama-2 without further training, outperforms Llama Guard, despite its extensive finetuning with a large dataset, in detecting jailbreak prompts. This superior performance is consistent across both zero-shot and adaptation scenarios, as evidenced by our evaluations on ToxicChat and XSTest. The source code is available at https://github.com/xyq7/GradSafe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19495v1">Qiskit Code Assistant: Training LLMs for generating Quantum Computing Code</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      Code Large Language Models (Code LLMs) have emerged as powerful tools, revolutionizing the software development landscape by automating the coding process and reducing time and effort required to build applications. This paper focuses on training Code LLMs to specialize in the field of quantum computing. We begin by discussing the unique needs of quantum computing programming, which differ significantly from classical programming approaches or languages. A Code LLM specializing in quantum computing requires a foundational understanding of quantum computing and quantum information theory. However, the scarcity of available quantum code examples and the rapidly evolving field, which necessitates continuous dataset updates, present significant challenges. Moreover, we discuss our work on training Code LLMs to produce high-quality quantum code using the Qiskit library. This work includes an examination of the various aspects of the LLMs used for training and the specific training conditions, as well as the results obtained with our current models. To evaluate our models, we have developed a custom benchmark, similar to HumanEval, which includes a set of tests specifically designed for the field of quantum computing programming using Qiskit. Our findings indicate that our model outperforms existing state-of-the-art models in quantum computing tasks. We also provide examples of code suggestions, comparing our model to other relevant code LLMs. Finally, we introduce a discussion on the potential benefits of Code LLMs for quantum computing computational scientists, researchers, and practitioners. We also explore various features and future work that could be relevant in this context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09989v4">LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2024-05-29
      | 💬 Accepted to Findings of ACL 2024
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) is a nascent multimodal task that aims to identify named entities, entity types and their corresponding visual regions. GMNER task exhibits two challenging properties: 1) The weak correlation between image-text pairs in social media results in a significant portion of named entities being ungroundable. 2) There exists a distinction between coarse-grained referring expressions commonly used in similar tasks (e.g., phrase localization, referring expression comprehension) and fine-grained named entities. In this paper, we propose RiVEG, a unified framework that reformulates GMNER into a joint MNER-VE-VG task by leveraging large language models (LLMs) as a connecting bridge. This reformulation brings two benefits: 1) It maintains the optimal MNER performance and eliminates the need for employing object detection methods to pre-extract regional features, thereby naturally addressing two major limitations of existing GMNER methods. 2) The introduction of entity expansion expression and Visual Entailment (VE) module unifies Visual Grounding (VG) and Entity Grounding (EG). It enables RiVEG to effortlessly inherit the Visual Entailment and Visual Grounding capabilities of any current or prospective multimodal pretraining models. Extensive experiments demonstrate that RiVEG outperforms state-of-the-art methods on the existing GMNER dataset and achieves absolute leads of 10.65%, 6.21%, and 8.83% in all three subtasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19164v1">Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery</a></div>
    <div class="paper-meta">
      📅 2024-05-29
      | 💬 8 pages, 2 tables, 6 figures
    </div>
    <details class="paper-abstract">
      Electronic Discovery (eDiscovery) involves identifying relevant documents from a vast collection based on legal production requests. The integration of artificial intelligence (AI) and natural language processing (NLP) has transformed this process, helping document review and enhance efficiency and cost-effectiveness. Although traditional approaches like BM25 or fine-tuned pre-trained models are common in eDiscovery, they face performance, computational, and interpretability challenges. In contrast, Large Language Model (LLM)-based methods prioritize interpretability but sacrifice performance and throughput. This paper introduces DISCOvery Graph (DISCOG), a hybrid approach that combines the strengths of two worlds: a heterogeneous graph-based method for accurate document relevance prediction and subsequent LLM-driven approach for reasoning. Graph representational learning generates embeddings and predicts links, ranking the corpus for a given request, and the LLMs provide reasoning for document relevance. Our approach handles datasets with balanced and imbalanced distributions, outperforming baselines in F1-score, precision, and recall by an average of 12%, 3%, and 16%, respectively. In an enterprise context, our approach drastically reduces document review costs by 99.9% compared to manual processes and by 95% compared to LLM-based classification methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.02179v2">Open-Source LLMs for Text Annotation: A Practical Guide for Model Setting and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      This paper studies the performance of open-source Large Language Models (LLMs) in text classification tasks typical for political science research. By examining tasks like stance, topic, and relevance classification, we aim to guide scholars in making informed decisions about their use of LLMs for text analysis. Specifically, we conduct an assessment of both zero-shot and fine-tuned LLMs across a range of text annotation tasks using news articles and tweets datasets. Our analysis shows that fine-tuning improves the performance of open-source LLMs, allowing them to match or even surpass zero-shot GPT-3.5 and GPT-4, though still lagging behind fine-tuned GPT-3.5. We further establish that fine-tuning is preferable to few-shot training with a relatively modest quantity of annotated text. Our findings show that fine-tuned open-source LLMs can be effectively deployed in a broad spectrum of text annotation applications. We provide a Python notebook facilitating the application of LLMs in text annotation for other researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19011v1">Examining the development of attitude scales using Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      For nearly a century, social researchers and psychologists have debated the efficacy of psychometric scales for attitude measurement, focusing on Thurstone's equal appearing interval scales and Likert's summated rating scales. Thurstone scales fell out of favour due to the labour intensive process of gathering judges' opinions on the initial items. However, advancements in technology have mitigated these challenges, nullifying the simplicity advantage of Likert scales, which have their own methodological issues. This study explores a methodological experiment to develop a Thurstone scale for assessing attitudes towards individuals living with AIDS. An electronic questionnaire was distributed to a group of judges, including undergraduate, postgraduate, and PhD students from disciplines such as social policy, law, medicine, and computer engineering, alongside established social researchers, and their responses were statistically analysed. The primary innovation of this study is the incorporation of an Artificial Intelligence (AI) Large Language Model (LLM) to evaluate the initial 63 items, comparing its assessments with those of the human judges. Interestingly, the AI provided also detailed explanations for its categorisation. Results showed no significant difference between AI and human judges for 35 items, minor differences for 23 items, and major differences for 5 items. This experiment demonstrates the potential of integrating AI with traditional psychometric methods to enhance the development of attitude measurement scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16899v1">Prompt-based vs. Fine-tuned LLMs Toward Causal Graph Verification</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      This work aims toward an application of natural language processing (NLP) technology for automatic verification of causal graphs using text sources. A causal graph is often derived from unsupervised causal discovery methods and requires manual evaluation from human experts. NLP technologies, i.e., Large Language Models (LLMs) such as BERT and ChatGPT, can potentially be used to verify the resulted causal graph by predicting if causal relation can be observed between node pairs based on the textual context. In this work, we compare the performance of two types of NLP models: (1) Pre-trained language models fine-tuned for causal relation classification task and, (2) prompt-based LLMs. Contrasted to previous studies where prompt-based LLMs work relatively well over a set of diverse tasks, preliminary experiments on biomedical and open-domain datasets suggest that the fine-tuned models far outperform the prompt-based LLMs, up to 20.5 points improvement of F1 score. We shared the code and the pre-processed datasets in our repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16107v6">Knowledge Fusion of Chat LLMs: A Preliminary Technical Report</a></div>
    <div class="paper-meta">
      📅 2024-05-29
      | 💬 Technical Report, work in progress
    </div>
    <details class="paper-abstract">
      Recently, FuseLLM introduced the concept of knowledge fusion to transfer the collective knowledge of multiple structurally varied LLMs into a target LLM through lightweight continual training. In this report, we extend the scalability and flexibility of the FuseLLM framework to realize the fusion of chat LLMs, resulting in FusionChat. FusionChat comprises two main stages. Firstly, we undertake knowledge fusion for structurally and scale-varied source LLMs to derive multiple target LLMs of identical structure and size via lightweight fine-tuning. Then, these target LLMs are merged within the parameter space, wherein we propose a novel method for determining the merging weights based on the variation ratio of parameter matrices before and after fine-tuning. We validate our approach using three prominent chat LLMs with diverse architectures and scales, namely NH2-Mixtral-8x7B, NH2-Solar-10.7B, and OpenChat-3.5-7B. Experimental results spanning various chat domains demonstrate the superiority of FusionChat-7B across a broad spectrum of chat LLMs at 7B and 34B scales, even surpassing GPT-3.5 (March) and approaching Mixtral-8x7B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18710v1">To FP8 and Back Again: Quantifying the Effects of Reducing Precision on LLM Training Stability</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      The massive computational costs associated with large language model (LLM) pretraining have spurred great interest in reduced-precision floating-point representations to accelerate the process. As a result, the BrainFloat16 (BF16) precision has become the de facto standard for LLM training, with hardware support included in recent accelerators. This trend has gone even further in the latest processors, where FP8 has recently been introduced. However, prior experience with FP16, which was found to be less stable than BF16, raises concerns as to whether FP8, with even fewer bits than FP16, can be a cost-effective option for LLM training. We argue that reduced-precision training schemes must have similar training stability and hyperparameter sensitivities to their higher-precision counterparts in order to be cost-effective. However, we find that currently available methods for FP8 training are not robust enough to allow their use as economical replacements. This prompts us to investigate the stability of reduced-precision LLM training in terms of robustness across random seeds and learning rates. To this end, we propose new evaluation techniques and a new metric for quantifying loss landscape sharpness in autoregressive language models. By simulating incremental bit reductions in floating-point representations, we analyze the relationship between representational power and training stability with the intent of aiding future research into the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18649v1">Training LLMs to Better Self-Debug and Explain Code</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      In the domain of code generation, self-debugging is crucial. It allows LLMs to refine their generated code based on execution feedback. This is particularly important because generating correct solutions in one attempt proves challenging for complex tasks. Prior works on self-debugging mostly focus on prompting methods by providing LLMs with few-shot examples, which work poorly on small open-sourced LLMs. In this work, we propose a training framework that significantly improves self-debugging capability of LLMs. Intuitively, we observe that a chain of explanations on the wrong code followed by code refinement helps LLMs better analyze the wrong code and do refinement. We thus propose an automated pipeline to collect a high-quality dataset for code explanation and refinement by generating a number of explanations and refinement trajectories and filtering via execution verification. We perform supervised fine-tuning (SFT) and further reinforcement learning (RL) on both success and failure trajectories with a novel reward design considering code explanation and refinement quality. SFT improves the pass@1 by up to 15.92% and pass@10 by 9.30% over four benchmarks. RL training brings additional up to 3.54% improvement on pass@1 and 2.55% improvement on pass@10. The trained LLMs show iterative refinement ability, and can keep refining code continuously. Lastly, our human evaluation shows that the LLMs trained with our framework generate more useful code explanations and help developers better understand bugs in source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18572v1">Low-rank finetuning for LLMs: A fairness perspective</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      Low-rank approximation techniques have become the de facto standard for fine-tuning Large Language Models (LLMs) due to their reduced computational and memory requirements. This paper investigates the effectiveness of these methods in capturing the shift of fine-tuning datasets from the initial pre-trained data distribution. Our findings reveal that there are cases in which low-rank fine-tuning falls short in learning such shifts. This, in turn, produces non-negligible side effects, especially when fine-tuning is adopted for toxicity mitigation in pre-trained models, or in scenarios where it is important to provide fair models. Through comprehensive empirical evidence on several models, datasets, and tasks, we show that low-rank fine-tuning inadvertently preserves undesirable biases and toxic behaviors. We also show that this extends to sequential decision-making tasks, emphasizing the need for careful evaluation to promote responsible LLMs development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05015v2">A Sober Look at LLMs for Material Discovery: Are They Actually Good for Bayesian Optimization Over Molecules?</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 ICML 2024. Code: https://github.com/wiseodd/lapeft-bayesopt
    </div>
    <details class="paper-abstract">
      Automation is one of the cornerstones of contemporary material discovery. Bayesian optimization (BO) is an essential part of such workflows, enabling scientists to leverage prior domain knowledge into efficient exploration of a large molecular space. While such prior knowledge can take many forms, there has been significant fanfare around the ancillary scientific knowledge encapsulated in large language models (LLMs). However, existing work thus far has only explored LLMs for heuristic materials searches. Indeed, recent work obtains the uncertainty estimate -- an integral part of BO -- from point-estimated, non-Bayesian LLMs. In this work, we study the question of whether LLMs are actually useful to accelerate principled Bayesian optimization in the molecular space. We take a sober, dispassionate stance in answering this question. This is done by carefully (i) viewing LLMs as fixed feature extractors for standard but principled BO surrogate models and by (ii) leveraging parameter-efficient finetuning methods and Bayesian neural networks to obtain the posterior of the LLM surrogate. Our extensive experiments with real-world chemistry problems show that LLMs can be useful for BO over molecules, but only if they have been pretrained or finetuned with domain-specific data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18361v1">Is a 3D-Tokenized LLM the Key to Reliable Autonomous Driving?</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      Rapid advancements in Autonomous Driving (AD) tasks turned a significant shift toward end-to-end fashion, particularly in the utilization of vision-language models (VLMs) that integrate robust logical reasoning and cognitive abilities to enable comprehensive end-to-end planning. However, these VLM-based approaches tend to integrate 2D vision tokenizers and a large language model (LLM) for ego-car planning, which lack 3D geometric priors as a cornerstone of reliable planning. Naturally, this observation raises a critical concern: Can a 2D-tokenized LLM accurately perceive the 3D environment? Our evaluation of current VLM-based methods across 3D object detection, vectorized map construction, and environmental caption suggests that the answer is, unfortunately, NO. In other words, 2D-tokenized LLM fails to provide reliable autonomous driving. In response, we introduce DETR-style 3D perceptrons as 3D tokenizers, which connect LLM with a one-layer linear projector. This simple yet elegant strategy, termed Atlas, harnesses the inherent priors of the 3D physical world, enabling it to simultaneously process high-resolution multi-view images and employ spatiotemporal modeling. Despite its simplicity, Atlas demonstrates superior performance in both 3D detection and ego planning tasks on nuScenes dataset, proving that 3D-tokenized LLM is the key to reliable autonomous driving. The code and datasets will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18359v1">Bridging the Gap: Dynamic Learning Strategies for Improving Multilingual Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 arXiv admin note: text overlap with arXiv:2305.17740
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are at the forefront of transforming numerous domains globally. However, their inclusivity and effectiveness remain limited for non-Latin scripts and low-resource languages. This paper tackles the imperative challenge of enhancing the multilingual performance of LLMs without extensive training or fine-tuning. Through systematic investigation and evaluation of diverse languages using popular question-answering (QA) datasets, we present novel techniques that unlock the true potential of LLMs in a polyglot landscape. Our approach encompasses three key strategies that yield significant improvements in multilingual proficiency. First, by meticulously optimizing prompts tailored for polyglot LLMs, we unlock their latent capabilities, resulting in substantial performance boosts across languages. Second, we introduce a new hybrid approach that synergizes LLM Retrieval Augmented Generation (RAG) with multilingual embeddings and achieves improved multilingual task performance. Finally, we introduce a novel learning approach that dynamically selects the optimal prompt strategy, LLM model, and embedding model per query at run-time. This dynamic adaptation maximizes the efficacy of LLMs across languages, outperforming best static and random strategies. Additionally, our approach adapts configurations in both offline and online settings, and can seamlessly adapt to new languages and datasets, leading to substantial advancements in multilingual understanding and generation across diverse languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04617v2">InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a cornerstone in real-world applications with lengthy streaming inputs (e.g., LLM-driven agents). However, existing LLMs, pre-trained on sequences with a restricted maximum length, cannot process longer sequences due to the out-of-domain and distraction issues. Common solutions often involve continual pre-training on longer sequences, which will introduce expensive computational overhead and uncontrollable change in model capabilities. In this paper, we unveil the intrinsic capacity of LLMs for understanding extremely long sequences without any fine-tuning. To this end, we introduce a training-free memory-based method, InfLLM. Specifically, InfLLM stores distant contexts into additional memory units and employs an efficient mechanism to lookup token-relevant units for attention computation. Thereby, InfLLM allows LLMs to efficiently process long sequences with a limited context window and well capture long-distance dependencies. Without any training, InfLLM enables LLMs that are pre-trained on sequences consisting of a few thousand tokens to achieve comparable performance with competitive baselines that continually train these LLMs on long sequences. Even when the sequence length is scaled to $1,024$K, InfLLM still effectively captures long-distance dependencies. Our code can be found in \url{https://github.com/thunlp/InfLLM}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09179v3">Instruction Backdoor Attacks Against Customized LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs development guidelines. We conduct extensive experiments on 6 prominent LLMs and 5 benchmark text classification datasets. The results show that our instruction backdoor attacks achieve the desired attack performance without compromising utility. Additionally, we propose two defense strategies and demonstrate their effectiveness in reducing such attacks. Our findings highlight the vulnerability and the potential risks of LLM customization such as GPTs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17789v1">Spanish and LLM Benchmarks: is MMLU Lost in Translation?</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) is a key element in their continuous improvement process and many benchmarks have been developed to assess the performance of LLMs in different tasks and topics. As LLMs become adopted worldwide, evaluating them in languages other than English is increasingly important. However, most LLM benchmarks are simply translated using an automated tool and then run in the target language. This means that the results depend not only on the LLM performance in that language but also on the quality of the translation. In this paper, we consider the case of the well-known Massive Multitask Language Understanding (MMLU) benchmark. Selected categories of the benchmark are translated into Spanish using Azure Translator and ChatGPT4 and run on ChatGPT4. Next, the results are processed to identify the test items that produce different answers in Spanish and English. Those are then analyzed manually to understand if the automatic translation caused the change. The results show that a significant fraction of the failing items can be attributed to mistakes in the translation of the benchmark. These results make a strong case for improving benchmarks in languages other than English by at least revising the translations of the items and preferably by adapting the tests to the target language by experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17893v1">Arithmetic Reasoning with LLM: Prolog Generation & Permutation</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 12 pages, 4 figures, accepted by NAACL 2024 Main Conference
    </div>
    <details class="paper-abstract">
      Instructing large language models (LLMs) to solve elementary school math problems has shown great success using Chain of Thought (CoT). However, the CoT approach relies on an LLM to generate a sequence of arithmetic calculations which can be prone to cascaded calculation errors. We hypothesize that an LLM should focus on extracting predicates and generating symbolic formulas from the math problem description so that the underlying calculation can be done via an external code interpreter. We investigate using LLM to generate Prolog programs to solve mathematical questions. Experimental results show that our Prolog-based arithmetic problem-solving outperforms CoT generation in the GSM8K benchmark across three distinct LLMs. In addition, given the insensitive ordering of predicates and symbolic formulas in Prolog, we propose to permute the ground truth predicates for more robust LLM training via data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17846v1">Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      Safety limitations in service robotics across various industries have raised significant concerns about the need for robust mechanisms ensuring that robots adhere to safe practices, thereby preventing actions that might harm humans or cause property damage. Despite advances, including the integration of Knowledge Graphs (KGs) with Large Language Models (LLMs), challenges in ensuring consistent safety in autonomous robot actions persist. In this paper, we propose a novel integration of Large Language Models with Embodied Robotic Control Prompts (ERCPs) and Embodied Knowledge Graphs (EKGs) to enhance the safety framework for service robots. ERCPs are designed as predefined instructions that ensure LLMs generate safe and precise responses. These responses are subsequently validated by EKGs, which provide a comprehensive knowledge base ensuring that the actions of the robot are continuously aligned with safety protocols, thereby promoting safer operational practices in varied contexts. Our experimental setup involved diverse real-world tasks, where robots equipped with our framework demonstrated significantly higher compliance with safety standards compared to traditional methods. This integration fosters secure human-robot interactions and positions our methodology at the forefront of AI-driven safety innovations in service robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03198v1">The Impossibility of Fair LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 Presented at the 1st Human-Centered Evaluation and Auditing of Language Models (HEAL) workshop at CHI 2024
    </div>
    <details class="paper-abstract">
      The need for fair AI is increasingly clear in the era of general-purpose systems such as ChatGPT, Gemini, and other large language models (LLMs). However, the increasing complexity of human-AI interaction and its social impacts have raised questions of how fairness standards could be applied. Here, we review the technical frameworks that machine learning researchers have used to evaluate fairness, such as group fairness and fair representations, and find that their application to LLMs faces inherent limitations. We show that each framework either does not logically extend to LLMs or presents a notion of fairness that is intractable for LLMs, primarily due to the multitudes of populations affected, sensitive attributes, and use cases. To address these challenges, we develop guidelines for the more realistic goal of achieving fairness in particular use cases: the criticality of context, the responsibility of LLM developers, and the need for stakeholder participation in an iterative process of design and evaluation. Moreover, it may eventually be possible and even necessary to use the general-purpose capabilities of AI systems to address fairness challenges as a form of scalable AI-assisted alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11592v3">Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      In the evolving landscape of natural language processing (NLP), fine-tuning pre-trained Large Language Models (LLMs) with first-order (FO) optimizers like SGD and Adam has become standard. Yet, as LLMs grow {in size}, the substantial memory overhead from back-propagation (BP) for FO gradient computation presents a significant challenge. Addressing this issue is crucial, especially for applications like on-device training where memory efficiency is paramount. This paper proposes a shift towards BP-free, zeroth-order (ZO) optimization as a solution for reducing memory costs during LLM fine-tuning, building on the initial concept introduced by MeZO. Unlike traditional ZO-SGD methods, our work expands the exploration to a wider array of ZO optimization techniques, through a comprehensive, first-of-its-kind benchmarking study across five LLM families (Roberta, OPT, LLaMA, Vicuna, Mistral), three task complexities, and five fine-tuning schemes. Our study unveils previously overlooked optimization principles, highlighting the importance of task alignment, the role of the forward gradient method, and the balance between algorithm complexity and fine-tuning performance. We further introduce novel enhancements to ZO optimization, including block-wise descent, hybrid training, and gradient sparsity. Our study offers a promising direction for achieving further memory-efficient LLM fine-tuning. Codes to reproduce all our experiments are at https://github.com/ZO-Bench/ZO-LLM .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17755v1">XL3M: A Training-free Framework for LLM Length Extension Based on Segment-wise Inference</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Length generalization failure problem, namely the large language model (LLM) fails to generalize to texts longer than its maximum training length, greatly restricts the application of LLM in the scenarios with streaming long inputs. To address this problem, the existing methods either require substantial costs or introduce precision loss. In this paper, we empirically find that the accuracy of the LLM's prediction is highly correlated to its certainty. Based on this, we propose an efficient training free framework, named XL3M (it means extra-long large language model), which enables the LLMs trained on short sequences to reason extremely long sequence without any further training or fine-tuning. Under the XL3M framework, the input context will be firstly decomposed into multiple short sub-contexts, where each sub-context contains an independent segment and a common ``question'' which is a few tokens from the end of the original context. Then XL3M gives a method to measure the relevance between each segment and the ``question'', and constructs a concise key context by splicing all the relevant segments in chronological order. The key context is further used instead of the original context to complete the inference task. Evaluations on comprehensive benchmarks show the superiority of XL3M. Using our framework, a Llama2-7B model is able to reason 20M long sequences on an 8-card Huawei Ascend 910B NPU machine with 64GB memory per card.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17741v1">LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      Recent literature has found that an effective method to customize or further improve large language models (LLMs) is to add dynamic adapters, such as low-rank adapters (LoRA) with Mixture-of-Experts (MoE) structures. Though such dynamic adapters incur modest computational complexity, they surprisingly lead to huge inference latency overhead, slowing down the decoding speed by 2.5+ times. In this paper, we analyze the fine-grained costs of the dynamic adapters and find that the fragmented CUDA kernel calls are the root cause. Therefore, we propose LoRA-Switch, a system-algorithm co-designed architecture for efficient dynamic adapters. Unlike most existing dynamic structures that adopt layer-wise or block-wise dynamic routing, LoRA-Switch introduces a token-wise routing mechanism. It switches the LoRA adapters and weights for each token and merges them into the backbone for inference. For efficiency, this switching is implemented with an optimized CUDA kernel, which fuses the merging operations for all LoRA adapters at once. Based on experiments with popular open-source LLMs on common benchmarks, our approach has demonstrated similar accuracy improvement as existing dynamic adapters, while reducing the decoding latency by more than 2.4 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16376v2">STRIDE: A Tool-Assisted LLM Agent Framework for Strategic and Interactive Decision-Making</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 39 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like GPT-4 have revolutionized natural language processing, showing remarkable linguistic proficiency and reasoning capabilities. However, their application in strategic multi-agent decision-making environments is hampered by significant limitations including poor mathematical reasoning, difficulty in following instructions, and a tendency to generate incorrect information. These deficiencies hinder their performance in strategic and interactive tasks that demand adherence to nuanced game rules, long-term planning, exploration in unknown environments, and anticipation of opponents' moves. To overcome these obstacles, this paper presents a novel LLM agent framework equipped with memory and specialized tools to enhance their strategic decision-making capabilities. We deploy the tools in a number of economically important environments, in particular bilateral bargaining and multi-agent and dynamic mechanism design. We employ quantitative metrics to assess the framework's performance in various strategic decision-making problems. Our findings establish that our enhanced framework significantly improves the strategic decision-making capability of LLMs. While we highlight the inherent limitations of current LLM models, we demonstrate the improvements through targeted enhancements, suggesting a promising direction for future developments in LLM applications for interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10958v2">Relative Preference Optimization: Enhancing LLM Alignment through Contrasting Responses across Identical and Diverse Prompts</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      In the field of large language models (LLMs), aligning models with the diverse preferences of users is a critical challenge. Direct Preference Optimization (DPO) has played a key role in this area. It works by using pairs of preferences derived from the same prompts, and it functions without needing an additional reward model. However, DPO does not fully reflect the complex nature of human learning, which often involves understanding contrasting responses to not only identical but also similar questions. To overcome this shortfall, we propose Relative Preference Optimization (RPO). RPO is designed to discern between more and less preferred responses derived from both identical and related prompts. It introduces a contrastive weighting mechanism, enabling the tuning of LLMs using a broader range of preference data, including both paired and unpaired sets. This approach expands the learning capabilities of the model, allowing it to leverage insights from a more varied set of prompts. Through empirical tests, including dialogue and summarization tasks, and evaluations using the AlpacaEval2.0 leaderboard, RPO has demonstrated a superior ability to align LLMs with user preferences and to improve their adaptability during the training process. Our code can be viewed at https://github.com/yinyueqin/relative-preference-optimization
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10379v2">DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 Published in ACL 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become a dominant and important tool for NLP researchers in a wide range of tasks. Today, many researchers use LLMs in synthetic data generation, task evaluation, fine-tuning, distillation, and other model-in-the-loop research workflows. However, challenges arise when using these models that stem from their scale, their closed source nature, and the lack of standardized tooling for these new and emerging workflows. The rapid rise to prominence of these models and these unique challenges has had immediate adverse impacts on open science and on the reproducibility of work that uses them. In this paper, we introduce DataDreamer, an open source Python library that allows researchers to write simple code to implement powerful LLM workflows. DataDreamer also helps researchers adhere to best practices that we propose to encourage open science and reproducibility. The library and documentation are available at https://github.com/datadreamer-dev/DataDreamer .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17386v1">MindMerger: Efficient Boosting LLM Reasoning in non-English Languages</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      Reasoning capabilities are crucial for Large Language Models (LLMs), yet a notable gap exists between English and non-English languages. To bridge this disparity, some works fine-tune LLMs to relearn reasoning capabilities in non-English languages, while others replace non-English inputs with an external model's outputs such as English translation text to circumvent the challenge of LLM understanding non-English. Unfortunately, these methods often underutilize the built-in skilled reasoning and useful language understanding capabilities of LLMs. In order to better utilize the minds of reasoning and language understanding in LLMs, we propose a new method, namely MindMerger, which merges LLMs with the external language understanding capabilities from multilingual models to boost the multilingual reasoning performance. Furthermore, a two-step training scheme is introduced to first train to embeded the external capabilities into LLMs and then train the collaborative utilization of the external capabilities and the built-in capabilities in LLMs. Experiments on three multilingual reasoning datasets and a language understanding dataset demonstrate that MindMerger consistently outperforms all baselines, especially in low-resource languages. Without updating the parameters of LLMs, the average accuracy improved by 6.7% and 8.0% across all languages and low-resource languages on the MGSM dataset, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17378v1">RTL-Repo: A Benchmark for Evaluating LLMs on Large-Scale RTL Design Projects</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated potential in assisting with Register Transfer Level (RTL) design tasks. Nevertheless, there remains to be a significant gap in benchmarks that accurately reflect the complexity of real-world RTL projects. To address this, this paper presents RTL-Repo, a benchmark specifically designed to evaluate LLMs on large-scale RTL design projects. RTL-Repo includes a comprehensive dataset of more than 4000 Verilog code samples extracted from public GitHub repositories, with each sample providing the full context of the corresponding repository. We evaluate several state-of-the-art models on the RTL-Repo benchmark, including GPT-4, GPT-3.5, Starcoder2, alongside Verilog-specific models like VeriGen and RTLCoder, and compare their performance in generating Verilog code for complex projects. The RTL-Repo benchmark provides a valuable resource for the hardware design community to assess and compare LLMs' performance in real-world RTL design scenarios and train LLMs specifically for Verilog code generation in complex, multi-file RTL projects. RTL-Repo is open-source and publicly available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00037v1">Aligning LLMs through Multi-perspective User Preference Ranking-based Feedback for Programming Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      Code Community Question Answering (CCQA) seeks to tackle programming-related issues, thereby boosting productivity in both software engineering and academic research. Recent advancements in Reinforcement Learning from Human Feedback (RLHF) have transformed the fine-tuning process of Large Language Models (LLMs) to produce responses that closely mimic human behavior. Leveraging LLMs with RLHF for practical CCQA applications has thus emerged as a promising area of study. Unlike standard code question-answering tasks, CCQA involves multiple possible answers, with varying user preferences for each response. Additionally, code communities often show a preference for new APIs. These challenges prevent LLMs from generating responses that cater to the diverse preferences of users in CCQA tasks. To address these issues, we propose a novel framework called Aligning LLMs through Multi-perspective User Preference Ranking-based Feedback for Programming Question Answering (ALMupQA) to create user-focused responses. Our approach starts with Multi-perspective Preference Ranking Alignment (MPRA), which synthesizes varied user preferences based on the characteristics of answers from code communities. We then introduce a Retrieval-augmented In-context Learning (RIL) module to mitigate the problem of outdated answers by retrieving responses to similar questions from a question bank. Due to the limited availability of high-quality, multi-answer CCQA datasets, we also developed a dataset named StaCCQA from real code communities. Extensive experiments demonstrated the effectiveness of the ALMupQA framework in terms of accuracy and user preference. Compared to the base model, ALMupQA showed nearly an 11% improvement in BLEU, with increases of 20% and 17.5% in BERTScore and CodeBERTScore, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17147v1">Large Language Models (LLMs): Deployment, Tokenomics and Sustainability</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 Accepted by IEEE CTSoc-NCT
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has significantly impacted human-computer interaction, epitomized by the release of GPT-4o, which introduced comprehensive multi-modality capabilities. In this paper, we first explored the deployment strategies, economic considerations, and sustainability challenges associated with the state-of-the-art LLMs. More specifically, we discussed the deployment debate between Retrieval-Augmented Generation (RAG) and fine-tuning, highlighting their respective advantages and limitations. After that, we quantitatively analyzed the requirement of xPUs in training and inference. Additionally, for the tokenomics of LLM services, we examined the balance between performance and cost from the quality of experience (QoE)'s perspective of end users. Lastly, we envisioned the future hybrid architecture of LLM processing and its corresponding sustainability concerns, particularly in the environmental carbon footprint impact. Through these discussions, we provided a comprehensive overview of the operational and strategic considerations essential for the responsible development and deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15238v2">GPT-HateCheck: Can LLMs Write Better Functional Tests for Hate Speech Detection?</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 LREC-COLING 2024. Content Warning: This paper contains model outputs that are offensive in nature
    </div>
    <details class="paper-abstract">
      Online hate detection suffers from biases incurred in data sampling, annotation, and model pre-training. Therefore, measuring the averaged performance over all examples in held-out test data is inadequate. Instead, we must identify specific model weaknesses and be informed when it is more likely to fail. A recent proposal in this direction is HateCheck, a suite for testing fine-grained model functionalities on synthesized data generated using templates of the kind "You are just a [slur] to me." However, despite enabling more detailed diagnostic insights, the HateCheck test cases are often generic and have simplistic sentence structures that do not match the real-world data. To address this limitation, we propose GPT-HateCheck, a framework to generate more diverse and realistic functional tests from scratch by instructing large language models (LLMs). We employ an additional natural language inference (NLI) model to verify the generations. Crowd-sourced annotation demonstrates that the generated test cases are of high quality. Using the new functional tests, we can uncover model weaknesses that would be overlooked using the original HateCheck dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02124v3">Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 ACL 2024 Main Conference. 64 pages (8 main), 70 figures, 37 tables. Blog: https://www.zjukg.org/project/MachineSoM
    </div>
    <details class="paper-abstract">
      As Natural Language Processing (NLP) systems are increasingly employed in intricate social environments, a pressing query emerges: Can these NLP systems mirror human-esque collaborative intelligence, in a multi-agent society consisting of multiple large language models (LLMs)? This paper probes the collaboration mechanisms among contemporary NLP systems by melding practical experiments with theoretical insights. We fabricate four unique `societies' comprised of LLM agents, where each agent is characterized by a specific `trait' (easy-going or overconfident) and engages in collaboration with a distinct `thinking pattern' (debate or reflection). Through evaluating these multi-agent societies on three benchmark datasets, we discern that certain collaborative strategies not only outshine previous top-tier approaches, but also optimize efficiency (using fewer API tokens). Moreover, our results further illustrate that LLM agents manifest human-like social behaviors, such as conformity and consensus reaching, mirroring foundational social psychology theories. In conclusion, we integrate insights from social psychology to contextualize the collaboration of LLM agents, inspiring further investigations into the collaboration mechanism for LLMs. We commit to sharing our code and datasets\footnote{\url{https://github.com/zjunlp/MachineSoM}.}, hoping to catalyze further research in this promising avenue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08699v2">Unsupervised Evaluation of Code LLMs with Round-Trip Correctness</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 Published in ICML 2024
    </div>
    <details class="paper-abstract">
      To evaluate code large language models (LLMs), research has relied on a few small manually curated benchmarks, such as HumanEval and MBPP, which represent a narrow part of the real-world software domains. In this work, we introduce round-trip correctness (RTC) as an alternative evaluation method. RTC allows Code LLM evaluation on a broader spectrum of real-world software domains without the need for costly human curation. RTC rests on the idea that we can ask a model to make a prediction (e.g., describe some code using natural language), feed that prediction back (e.g., synthesize code from the predicted description), and check if this round-trip leads to code that is semantically equivalent to the original input. We show how to employ RTC to evaluate code synthesis and editing. We find that RTC strongly correlates with model performance on existing narrow-domain code synthesis benchmarks while allowing us to expand to a much broader set of domains and tasks which was not previously possible without costly human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05445v2">Accurate LoRA-Finetuning Quantization of LLMs via Information Retention</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      The LoRA-finetuning quantization of LLMs has been extensively studied to obtain accurate yet compact LLMs for deployment on resource-constrained hardware. However, existing methods cause the quantized LLM to severely degrade and even fail to benefit from the finetuning of LoRA. This paper proposes a novel IR-QLoRA for pushing quantized LLMs with LoRA to be highly accurate through information retention. The proposed IR-QLoRA mainly relies on two technologies derived from the perspective of unified information: (1) statistics-based Information Calibration Quantization allows the quantized parameters of LLM to retain original information accurately; (2) finetuning-based Information Elastic Connection makes LoRA utilizes elastic representation transformation with diverse information. Comprehensive experiments show that IR-QLoRA can significantly improve accuracy across LLaMA and LLaMA2 families under 2-4 bit-widths, e.g., 4- bit LLaMA-7B achieves 1.4% improvement on MMLU compared with the state-of-the-art methods. The significant performance gain requires only a tiny 0.31% additional time consumption, revealing the satisfactory efficiency of our IR-QLoRA. We highlight that IR-QLoRA enjoys excellent versatility, compatible with various frameworks (e.g., NormalFloat and Integer quantization) and brings general accuracy gains. The code is available at https://github.com/htqin/ir-qlora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16856v1">Can We Trust LLMs? Mitigate Overconfidence Bias in LLMs through Knowledge Transfer</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      The study explores mitigating overconfidence bias in LLMs to improve their reliability. We introduce a knowledge transfer (KT) method utilizing chain of thoughts, where "big" LLMs impart knowledge to "small" LLMs via detailed, sequential reasoning paths. This method uses advanced reasoning of larger models to fine-tune smaller models, enabling them to produce more accurate predictions with calibrated confidence. Experimental evaluation using multiple-choice questions and sentiment analysis across diverse datasets demonstrated the KT method's superiority over the vanilla and question-answer pair (QA) fine-tuning methods. The most significant improvement in three key metrics, where the KT method outperformed the vanilla and QA methods by an average of 55.3% and 43.1%, respectively. These findings underscore the KT method's potential in enhancing model trustworthiness and accuracy, offering precise outputs with well-matched confidence levels across various contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11299v2">The CAP Principle for LLM Serving: A Survey of Long-Context Large Language Model Serving</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      We survey the large language model (LLM) serving area to understand the intricate dynamics between cost-efficiency and accuracy, which is magnified by the growing need for longer contextual understanding when deploying models at a massive scale. Our findings reveal that works in this space optimize along three distinct but conflicting goals: improving serving context length (C), improving serving accuracy (A), and improving serving performance (P). Drawing inspiration from the CAP theorem in databases, we propose a CAP principle for LLM serving, which suggests that any optimization can improve at most two of these three goals simultaneously. Our survey categorizes existing works within this framework. We find the definition and continuity of user-perceived measurement metrics are crucial in determining whether a goal has been met, akin to prior CAP databases in the wild. We recognize the CAP principle for LLM serving as a guiding principle, rather than a formal theorem, to inform designers of the inherent and dynamic trade-offs in serving models. As serving accuracy and performance have been extensively studied, this survey focuses on works that extend serving context length and address the resulting challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16714v1">Crafting Interpretable Embeddings by Asking LLMs Questions</a></div>
    <div class="paper-meta">
      📅 2024-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly improved text embeddings for a growing array of natural-language processing tasks. However, their opaqueness and proliferation into scientific domains such as neuroscience have created a growing need for interpretability. Here, we ask whether we can obtain interpretable embeddings through LLM prompting. We introduce question-answering embeddings (QA-Emb), embeddings where each feature represents an answer to a yes/no question asked to an LLM. Training QA-Emb reduces to selecting a set of underlying questions rather than learning model weights. We use QA-Emb to flexibly generate interpretable models for predicting fMRI voxel responses to language stimuli. QA-Emb significantly outperforms an established interpretable baseline, and does so while requiring very few questions. This paves the way towards building flexible feature spaces that can concretize and evaluate our understanding of semantic brain representations. We additionally find that QA-Emb can be effectively approximated with an efficient model, and we explore broader applications in simple NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14992v2">tinyBenchmarks: evaluating LLMs with fewer examples</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 Proceedings of the 41st International Conference on Machine Learning (ICML)
    </div>
    <details class="paper-abstract">
      The versatility of large language models (LLMs) led to the creation of diverse benchmarks that thoroughly test a variety of language models' abilities. These benchmarks consist of tens of thousands of examples making evaluation of LLMs very expensive. In this paper, we investigate strategies to reduce the number of evaluations needed to assess the performance of an LLM on several key benchmarks. For example, we show that to accurately estimate the performance of an LLM on MMLU, a popular multiple-choice QA benchmark consisting of 14K examples, it is sufficient to evaluate this LLM on 100 curated examples. We release evaluation tools and tiny versions of popular benchmarks: Open LLM Leaderboard, MMLU, HELM, and AlpacaEval 2.0. Our empirical analysis demonstrates that these tools and tiny benchmarks are sufficient to reliably and efficiently reproduce the original evaluation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00034v1">Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for Diverse Hallucinations Categories</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 arXiv admin note: text overlap with arXiv:2402.17811
    </div>
    <details class="paper-abstract">
      Recent studies have indicated that Large Language Models (LLMs) harbor an inherent understanding of truthfulness, yet often fail to express fully and generate false statements. This gap between "knowing" and "telling" poses a challenge for ensuring the truthfulness of generated content. To address this, we introduce Adaptive Activation Steering (ACT), a tuning-free method that adaptively shift LLM's activations in "truthful" direction during inference. ACT addresses diverse categories of hallucinations by utilizing diverse steering vectors and adjusting the steering intensity adaptively. Applied as an add-on across various models, ACT significantly improves truthfulness in LLaMA ($\uparrow$ 142\%), LLaMA2 ($\uparrow$ 24\%), Alpaca ($\uparrow$ 36\%), Vicuna ($\uparrow$ 28\%), and LLaMA2-Chat ($\uparrow$ 19\%). Furthermore, we verify ACT's scalability across larger models (13B, 33B, 65B), underscoring the adaptability of ACT to large-scale language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.12288v4">The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 21 pages, 11 figures
    </div>
    <details class="paper-abstract">
      We expose a surprising failure of generalization in auto-regressive large language models (LLMs). If a model is trained on a sentence of the form "A is B", it will not automatically generalize to the reverse direction "B is A". This is the Reversal Curse. For instance, if a model is trained on "Valentina Tereshkova was the first woman to travel to space", it will not automatically be able to answer the question, "Who was the first woman to travel to space?". Moreover, the likelihood of the correct answer ("Valentina Tershkova") will not be higher than for a random name. Thus, models do not generalize a prevalent pattern in their training set: if "A is B" occurs, "B is A" is more likely to occur. It is worth noting, however, that if "A is B" appears in-context, models can deduce the reverse relationship. We provide evidence for the Reversal Curse by finetuning GPT-3 and Llama-1 on fictitious statements such as "Uriah Hawthorne is the composer of Abyssal Melodies" and showing that they fail to correctly answer "Who composed Abyssal Melodies?". The Reversal Curse is robust across model sizes and model families and is not alleviated by data augmentation. We also evaluate ChatGPT (GPT-3.5 and GPT-4) on questions about real-world celebrities, such as "Who is Tom Cruise's mother? [A: Mary Lee Pfeiffer]" and the reverse "Who is Mary Lee Pfeiffer's son?". GPT-4 correctly answers questions like the former 79% of the time, compared to 33% for the latter. Code available at: https://github.com/lukasberglund/reversal_curse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18659v2">DetermLR: Augmenting LLM-based Logical Reasoning from Indeterminacy to Determinacy</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 Accepted at ACL 2024 Main, Code repo: https://github.com/XiaoMi/DetermLR
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have revolutionized the landscape of reasoning tasks. To enhance the capabilities of LLMs to emulate human reasoning, prior studies have focused on modeling reasoning steps using various thought structures like chains, trees, or graphs. However, LLM-based reasoning still encounters the following challenges: (1) Limited adaptability of preset structures to diverse tasks; (2) Insufficient precision in exploiting known conditions to derive new ones; and (3) Inadequate consideration of historical reasoning experiences for subsequent reasoning steps. To this end, we propose DetermLR, a novel perspective that rethinks the reasoning process as an evolution from indeterminacy to determinacy. First, we categorize known conditions into two types: determinate and indeterminate premises This provides an oveall direction for the reasoning process and guides LLMs in converting indeterminate data into progressively determinate insights. Subsequently, we leverage quantitative measurements to prioritize more relevant premises to explore new insights. Furthermore, we automate the storage and extraction of available premises and reasoning paths with reasoning memory, preserving historical reasoning details for subsequent reasoning steps. Comprehensive experimental results demonstrate that DetermLR surpasses all baselines on various logical reasoning benchmarks: LogiQA, ProofWriter, FOLIO, PrOntoQA, and LogicalDeduction. Compared to previous multi-step reasoning methods, DetermLR achieves higher accuracy with fewer reasoning steps, highlighting its superior efficiency and effectiveness in solving logical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16579v1">Automatically Generating Numerous Context-Driven SFT Data for LLMs across Diverse Granularity</a></div>
    <div class="paper-meta">
      📅 2024-05-26
    </div>
    <details class="paper-abstract">
      Constructing high-quality query-response pairs from custom corpus is crucial for supervised fine-tuning (SFT) large language models (LLMs) in many applications, like creating domain-specific AI assistants or roleplaying agents. However, sourcing this data through human annotation is costly, and existing automated methods often fail to capture the diverse range of contextual granularity and tend to produce homogeneous data. To tackle these issues, we introduce a novel method named AugCon, capable of automatically generating context-driven SFT data across multiple levels of granularity with high diversity, quality and fidelity. AugCon begins by generating queries using the Context-Split-Tree (CST), an innovative approach for recursively deriving queries and splitting context to cover full granularity. Then, we train a scorer through contrastive learning to collaborate with CST to rank and refine queries. Finally, a synergistic integration of self-alignment and self-improving is introduced to obtain high-fidelity responses. Extensive experiments are conducted incorporating both human and automatic evaluations, encompassing a test scenario and four widely-used benchmarks in English and Chinese. The results highlight the significant advantages of AugCon in producing high diversity, quality, and fidelity SFT data against several state-of-the-art methods. All of our code, dataset, and fine-tuned model will be available at: https://github.com/quanshr/AugCon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11977v1">Building Better AI Agents: A Provocation on the Utilisation of Persona in LLM-based Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 Accepted by The international ACM Conversational User Interfaces (CUI) conference 2024
    </div>
    <details class="paper-abstract">
      The incorporation of Large Language Models (LLMs) such as the GPT series into diverse sectors including healthcare, education, and finance marks a significant evolution in the field of artificial intelligence (AI). The increasing demand for personalised applications motivated the design of conversational agents (CAs) to possess distinct personas. This paper commences by examining the rationale and implications of imbuing CAs with unique personas, smoothly transitioning into a broader discussion of the personalisation and anthropomorphism of CAs based on LLMs in the LLM era. We delve into the specific applications where the implementation of a persona is not just beneficial but critical for LLM-based CAs. The paper underscores the necessity of a nuanced approach to persona integration, highlighting the potential challenges and ethical dilemmas that may arise. Attention is directed towards the importance of maintaining persona consistency, establishing robust evaluation mechanisms, and ensuring that the persona attributes are effectively complemented by domain-specific knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05854v4">LTGC: Long-tail Recognition via Leveraging LLMs-driven Generated Content</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 CVPR 2024, Oral
    </div>
    <details class="paper-abstract">
      Long-tail recognition is challenging because it requires the model to learn good representations from tail categories and address imbalances across all categories. In this paper, we propose a novel generative and fine-tuning framework, LTGC, to handle long-tail recognition via leveraging generated content. Firstly, inspired by the rich implicit knowledge in large-scale models (e.g., large language models, LLMs), LTGC leverages the power of these models to parse and reason over the original tail data to produce diverse tail-class content. We then propose several novel designs for LTGC to ensure the quality of the generated data and to efficiently fine-tune the model using both the generated and original data. The visualization demonstrates the effectiveness of the generation module in LTGC, which produces accurate and diverse tail data. Additionally, the experimental results demonstrate that our LTGC outperforms existing state-of-the-art methods on popular long-tailed benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14314v2">Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Grounding the reasoning ability of large language models (LLMs) for embodied tasks is challenging due to the complexity of the physical world. Especially, LLM planning for multi-agent collaboration requires communication of agents or credit assignment as the feedback to re-adjust the proposed plans and achieve effective coordination. However, existing methods that overly rely on physical verification or self-reflection suffer from excessive and inefficient querying of LLMs. In this paper, we propose a novel framework for multi-agent collaboration that introduces Reinforced Advantage feedback (ReAd) for efficient self-refinement of plans. Specifically, we perform critic regression to learn a sequential advantage function from LLM-planned data, and then treat the LLM planner as an optimizer to generate actions that maximize the advantage function. It endows the LLM with the foresight to discern whether the action contributes to accomplishing the final task. We provide theoretical analysis by extending advantage-weighted regression in reinforcement learning to multi-agent systems. Experiments on Overcooked-AI and a difficult variant of RoCoBench show that ReAd surpasses baselines in success rate, and also significantly decreases the interaction steps of agents and query rounds of LLMs, demonstrating its high efficiency for grounding LLMs. More results are given at https://read-llm.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11000v2">OVAL-Prompt: Open-Vocabulary Affordance Localization for Robot Manipulation through LLM Affordance-Grounding</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 Accepted to Vision-Language Models for Navigation and Manipulation (VLMNM) Workshop (ICRA 2024)
    </div>
    <details class="paper-abstract">
      In order for robots to interact with objects effectively, they must understand the form and function of each object they encounter. Essentially, robots need to understand which actions each object affords, and where those affordances can be acted on. Robots are ultimately expected to operate in unstructured human environments, where the set of objects and affordances is not known to the robot before deployment (i.e. the open-vocabulary setting). In this work, we introduce OVAL-Prompt, a prompt-based approach for open-vocabulary affordance localization in RGB-D images. By leveraging a Vision Language Model (VLM) for open-vocabulary object part segmentation and a Large Language Model (LLM) to ground each part-segment-affordance, OVAL-Prompt demonstrates generalizability to novel object instances, categories, and affordances without domain-specific finetuning. Quantitative experiments demonstrate that without any finetuning, OVAL-Prompt achieves localization accuracy that is competitive with supervised baseline models. Moreover, qualitative experiments show that OVAL-Prompt enables affordance-based robot manipulation of open-vocabulary object instances and categories. Project Page: https://ekjt.github.io/OVAL-Prompt/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16311v1">Understanding Stakeholders' Perceptions and Needs Across the LLM Supply Chain</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 Paper accepted at the HCXAI workshop, co-located with CHI'24
    </div>
    <details class="paper-abstract">
      Explainability and transparency of AI systems are undeniably important, leading to several research studies and tools addressing them. Existing works fall short of accounting for the diverse stakeholders of the AI supply chain who may differ in their needs and consideration of the facets of explainability and transparency. In this paper, we argue for the need to revisit the inquiries of these vital constructs in the context of LLMs. To this end, we report on a qualitative study with 71 different stakeholders, where we explore the prevalent perceptions and needs around these concepts. This study not only confirms the importance of exploring the ``who'' in XAI and transparency for LLMs, but also reflects on best practices to do so while surfacing the often forgotten stakeholders and their information needs. Our insights suggest that researchers and practitioners should simultaneously clarify the ``who'' in considerations of explainability and transparency, the ``what'' in the information needs, and ``why'' they are needed to ensure responsible design and development across the LLM supply chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16310v1">An Empirical Exploration of Trust Dynamics in LLM Supply Chains</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 Paper accepted at the TREW workshop co-located with CHI'24
    </div>
    <details class="paper-abstract">
      With the widespread proliferation of AI systems, trust in AI is an important and timely topic to navigate. Researchers so far have largely employed a myopic view of this relationship. In particular, a limited number of relevant trustors (e.g., end-users) and trustees (i.e., AI systems) have been considered, and empirical explorations have remained in laboratory settings, potentially overlooking factors that impact human-AI relationships in the real world. In this paper, we argue for broadening the scope of studies addressing `trust in AI' by accounting for the complex and dynamic supply chains that AI systems result from. AI supply chains entail various technical artifacts that diverse individuals, organizations, and stakeholders interact with, in a variety of ways. We present insights from an in-situ, empirical study of LLM supply chains. Our work reveals additional types of trustors and trustees and new factors impacting their trust relationships. These relationships were found to be central to the development and adoption of LLMs, but they can also be the terrain for uncalibrated trust and reliance on untrustworthy LLMs. Based on these findings, we discuss the implications for research on `trust in AI'. We highlight new research opportunities and challenges concerning the appropriate study of inter-actor relationships across the supply chain and the development of calibrated trust and meaningful reliance behaviors. We also question the meaning of building trust in the LLM supply chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.04152v4">Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 Accepted by ICLR 2024 (Spotlight)
    </div>
    <details class="paper-abstract">
      Recent advancements in Multimodal Large Language Models (MLLMs) have been utilizing Visual Prompt Generators (VPGs) to convert visual features into tokens that LLMs can recognize. This is achieved by training the VPGs on millions of image-caption pairs, where the VPG-generated tokens of images are fed into a frozen LLM to generate the corresponding captions. However, this image-captioning based training objective inherently biases the VPG to concentrate solely on the primary visual contents sufficient for caption generation, often neglecting other visual details. This shortcoming results in MLLMs' underperformance in comprehending demonstrative instructions consisting of multiple, interleaved, and multimodal instructions that demonstrate the required context to complete a task. To address this issue, we introduce a generic and lightweight Visual Prompt Generator Complete module (VPG-C), which can infer and complete the missing details essential for comprehending demonstrative instructions. Further, we propose a synthetic discriminative training strategy to fine-tune VPG-C, eliminating the need for supervised demonstrative instructions. As for evaluation, we build DEMON, a comprehensive benchmark for demonstrative instruction understanding. Synthetically trained with the proposed strategy, VPG-C achieves significantly stronger zero-shot performance across all tasks of DEMON. Further evaluation on the MME and OwlEval benchmarks also demonstrate the superiority of VPG-C. Our benchmark, code, and pre-trained models are available at https://github.com/DCDmllm/Cheetah.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16241v1">FastQuery: Communication-efficient Embedding Table Query for Private LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 6 pages, DAC2024
    </div>
    <details class="paper-abstract">
      With the fast evolution of large language models (LLMs), privacy concerns with user queries arise as they may contain sensitive information. Private inference based on homomorphic encryption (HE) has been proposed to protect user query privacy. However, a private embedding table query has to be formulated as a HE-based matrix-vector multiplication problem and suffers from enormous computation and communication overhead. We observe the overhead mainly comes from the neglect of 1) the one-hot nature of user queries and 2) the robustness of the embedding table to low bit-width quantization noise. Hence, in this paper, we propose a private embedding table query optimization framework, dubbed FastQuery. FastQuery features a communication-aware embedding table quantization algorithm and a one-hot-aware dense packing algorithm to simultaneously reduce both the computation and communication costs. Compared to prior-art HE-based frameworks, e.g., Cheetah, Iron, and Bumblebee, FastQuery achieves more than $4.3\times$, $2.7\times$, $1.3\times$ latency reduction, respectively and more than $75.7\times$, $60.2\times$, $20.2\times$ communication reduction, respectively, on both LLAMA-7B and LLAMA-30B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16081v1">A Study on Developer Behaviors for Validating and Repairing LLM-Generated Code Using Eye Tracking and IDE Actions</a></div>
    <div class="paper-meta">
      📅 2024-05-25
    </div>
    <details class="paper-abstract">
      The increasing use of large language model (LLM)-powered code generation tools, such as GitHub Copilot, is transforming software engineering practices. This paper investigates how developers validate and repair code generated by Copilot and examines the impact of code provenance awareness during these processes. We conducted a lab study with 28 participants, who were tasked with validating and repairing Copilot-generated code in three software projects. Participants were randomly divided into two groups: one informed about the provenance of LLM-generated code and the other not. We collected data on IDE interactions, eye-tracking, cognitive workload assessments, and conducted semi-structured interviews. Our results indicate that, without explicit information, developers often fail to identify the LLM origin of the code. Developers generally employ similar validation and repair strategies for LLM-generated code, but exhibit behaviors such as frequent switching between code and comments, different attentional focus, and a tendency to delete and rewrite code. Being aware of the code's provenance led to improved performance, increased search efforts, more frequent Copilot usage, and higher cognitive workload. These findings enhance our understanding of how developers interact with LLM-generated code and carry implications for designing tools that facilitate effective human-LLM collaboration in software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16064v1">Keypoint-based Progressive Chain-of-Thought Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 Accepted by ICML 2024
    </div>
    <details class="paper-abstract">
      Chain-of-thought distillation is a powerful technique for transferring reasoning abilities from large language models (LLMs) to smaller student models. Previous methods typically require the student to mimic the step-by-step rationale produced by LLMs, often facing the following challenges: (i) Tokens within a rationale vary in significance, and treating them equally may fail to accurately mimic keypoint tokens, leading to reasoning errors. (ii) They usually distill knowledge by consistently predicting all the steps in a rationale, which falls short in distinguishing the learning order of step generation. This diverges from the human cognitive progression of starting with easy tasks and advancing to harder ones, resulting in sub-optimal outcomes. To this end, we propose a unified framework, called KPOD, to address these issues. Specifically, we propose a token weighting module utilizing mask learning to encourage accurate mimicry of keypoint tokens by the student during distillation. Besides, we develop an in-rationale progressive distillation strategy, starting with training the student to generate the final reasoning steps and gradually extending to cover the entire rationale. To accomplish this, a weighted token generation loss is proposed to assess step reasoning difficulty, and a value function is devised to schedule the progressive distillation by considering both step difficulty and question diversity. Extensive experiments on four reasoning benchmarks illustrate our KPOD outperforms previous methods by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15318v1">Are Long-LLMs A Necessity For Long-Context Tasks?</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      The learning and deployment of long-LLMs remains a challenging problem despite recent progresses. In this work, we argue that the long-LLMs are not a necessity to solve long-context tasks, as common long-context tasks are short-context solvable, i.e. they can be solved by purely working with oracle short-contexts within the long-context tasks' inputs. On top of this argument, we propose a framework called LC-Boost (Long-Context Bootstrapper), which enables a short-LLM to address the long-context tasks in a bootstrapping manner. In our framework, the short-LLM prompts itself to reason for two critical decisions: 1) how to access to the appropriate part of context within the input, 2) how to make effective use of the accessed context. By adaptively accessing and utilizing the context based on the presented tasks, LC-Boost can serve as a general framework to handle diversified long-context processing problems. We comprehensively evaluate different types of tasks from popular long-context benchmarks, where LC-Boost is able to achieve a substantially improved performance with a much smaller consumption of resource.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16910v2">NeSy is alive and well: A LLM-driven symbolic approach for better code comment data generation and classification</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 19 pages, 5 figures, accepted for the GeNeSy workshop at the Extended Semantic Web Conference (ESWC) 2024
    </div>
    <details class="paper-abstract">
      We present a neuro-symbolic (NeSy) workflow combining a symbolic-based learning technique with a large language model (LLM) agent to generate synthetic data for code comment classification in the C programming language. We also show how generating controlled synthetic data using this workflow fixes some of the notable weaknesses of LLM-based generation and increases the performance of classical machine learning models on the code comment classification task. Our best model, a Neural Network, achieves a Macro-F1 score of 91.412% with an increase of 1.033% after data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.05087v2">PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 Accepted by ICLR 2024
    </div>
    <details class="paper-abstract">
      Instruction tuning large language models (LLMs) remains a challenging task, owing to the complexity of hyperparameter selection and the difficulty involved in evaluating the tuned models. To determine the optimal hyperparameters, an automatic, robust, and reliable evaluation benchmark is essential. However, establishing such a benchmark is not a trivial task due to the challenges associated with evaluation accuracy and privacy protection. In response to these challenges, we introduce a judge large language model, named PandaLM, which is trained to distinguish the superior model given several LLMs. PandaLM's focus extends beyond just the objective correctness of responses, which is the main focus of traditional evaluation datasets. It addresses vital subjective factors such as relative conciseness, clarity, adherence to instructions, comprehensiveness, and formality. To ensure the reliability of PandaLM, we collect a diverse human-annotated test dataset, where all contexts are generated by humans and labels are aligned with human preferences. Our results indicate that PandaLM-7B achieves 93.75% of GPT-3.5's evaluation ability and 88.28% of GPT-4's in terms of F1-score on our test dataset. PandaLM enables the evaluation of LLM to be fairer but with less cost, evidenced by significant improvements achieved by models tuned through PandaLM compared to their counterparts trained with default Alpaca's hyperparameters. In addition, PandaLM does not depend on API-based evaluations, thus avoiding potential data leakage. All resources of PandaLM are released at https://github.com/WeOpenML/PandaLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15250v1">Coaching Copilot: Blended Form of an LLM-Powered Chatbot and a Human Coach to Effectively Support Self-Reflection for Leadership Growth</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 Accepted by the International ACM Conversational User Interfaces Conference (CUI '24)
    </div>
    <details class="paper-abstract">
      Chatbots' role in fostering self-reflection is now widely recognized, especially in inducing users' behavior change. While the benefits of 24/7 availability, scalability, and consistent responses have been demonstrated in contexts such as healthcare and tutoring to help one form a new habit, their utilization in coaching necessitating deeper introspective dialogue to induce leadership growth remains unexplored. This paper explores the potential of such a chatbot powered by recent Large Language Models (LLMs) in collaboration with professional coaches in the field of executive coaching. Through a design workshop with them and two weeks of user study involving ten coach-client pairs, we explored the feasibility and nuances of integrating chatbots to complement human coaches. Our findings highlight the benefits of chatbots' ubiquity and reasoning capabilities enabled by LLMs while identifying their limitations and design necessities for effective collaboration between human coaches and chatbots. By doing so, this work contributes to the foundation for augmenting one's self-reflective process with prevalent conversational agents through the human-in-the-loop approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06549v1">Large Language Model (LLM) for Standard Cell Layout Design Optimization</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 6 pages, 8 figures, IEEE International Workshop on LLM-Aided Design (LAD'24)
    </div>
    <details class="paper-abstract">
      Standard cells are essential components of modern digital circuit designs. With process technologies advancing toward 2nm, more routability issues have arisen due to the decreasing number of routing tracks, increasing number and complexity of design rules, and strict patterning rules. The state-of-the-art standard cell design automation framework is able to automatically design standard cell layouts in advanced nodes, but it is still struggling to generate highly competitive Performance-Power-Area (PPA) and routable cell layouts for complex sequential cell designs. Consequently, a novel and efficient methodology incorporating the expertise of experienced human designers to incrementally optimize the PPA of cell layouts is highly necessary and essential. High-quality device clustering, with consideration of netlist topology, diffusion sharing/break and routability in the layouts, can reduce complexity and assist in finding highly competitive PPA, and routable layouts faster. In this paper, we leverage the natural language and reasoning ability of Large Language Model (LLM) to generate high-quality cluster constraints incrementally to optimize the cell layout PPA and debug the routability with ReAct prompting. On a benchmark of sequential standard cells in 2nm, we demonstrate that the proposed method not only achieves up to 19.4% smaller cell area, but also generates 23.5% more LVS/DRC clean cell layouts than previous work. In summary, the proposed method not only successfully reduces cell area by 4.65% on average, but also is able to fix routability in the cell layout designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15202v1">Cross-Task Defense: Instruction-Tuning LLMs for Content Safety</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 accepted to NAACL2024 TrustNLP workshop
    </div>
    <details class="paper-abstract">
      Recent studies reveal that Large Language Models (LLMs) face challenges in balancing safety with utility, particularly when processing long texts for NLP tasks like summarization and translation. Despite defenses against malicious short questions, the ability of LLMs to safely handle dangerous long content, such as manuals teaching illicit activities, remains unclear. Our work aims to develop robust defenses for LLMs in processing malicious documents alongside benign NLP task queries. We introduce a defense dataset comprised of safety-related examples and propose single-task and mixed-task losses for instruction tuning. Our empirical results demonstrate that LLMs can significantly enhance their capacity to safely manage dangerous content with appropriate instruction tuning. Additionally, strengthening the defenses of tasks most susceptible to misuse is effective in protecting LLMs against processing harmful information. We also observe that trade-offs between utility and safety exist in defense strategies, where Llama2, utilizing our proposed approach, displays a significantly better balance compared to Llama1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15165v1">A Solution-based LLM API-using Methodology for Academic Information Seeking</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 22 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Applying large language models (LLMs) for academic API usage shows promise in reducing researchers' academic information seeking efforts. However, current LLM API-using methods struggle with complex API coupling commonly encountered in academic queries. To address this, we introduce SoAy, a solution-based LLM API-using methodology for academic information seeking. It uses code with a solution as the reasoning method, where a solution is a pre-constructed API calling sequence. The addition of the solution reduces the difficulty for the model to understand the complex relationships between APIs. Code improves the efficiency of reasoning. To evaluate SoAy, we introduce SoAyBench, an evaluation benchmark accompanied by SoAyEval, built upon a cloned environment of APIs from AMiner. Experimental results demonstrate a 34.58-75.99\% performance improvement compared to state-of-the-art LLM API-based baselines. All datasets, codes, tuned models, and deployed online services are publicly accessible at https://github.com/RUCKBReasoning/SoAy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15114v1">Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning</a></div>
    <div class="paper-meta">
      📅 2024-05-24
    </div>
    <details class="paper-abstract">
      Conventional recommender systems (RSs) face challenges in precisely capturing users' fine-grained preferences. Large language models (LLMs) have shown capabilities in commonsense reasoning and leveraging external tools that may help address these challenges. However, existing LLM-based RSs suffer from hallucinations, misalignment between the semantic space of items and the behavior space of users, or overly simplistic control strategies (e.g., whether to rank or directly present existing results). To bridge these gap, we introduce ToolRec, a framework for LLM-empowered recommendations via tool learning that uses LLMs as surrogate users, thereby guiding the recommendation process and invoking external tools to generate a recommendation list that aligns closely with users' nuanced preferences. We formulate the recommendation process as a process aimed at exploring user interests in attribute granularity. The process factors in the nuances of the context and user preferences. The LLM then invokes external tools based on a user's attribute instructions and probes different segments of the item pool. We consider two types of attribute-oriented tools: rank tools and retrieval tools. Through the integration of LLMs, ToolRec enables conventional recommender systems to become external tools with a natural language interface. Extensive experiments verify the effectiveness of ToolRec, particularly in scenarios that are rich in semantic content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15902v1">Hacc-Man: An Arcade Game for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-24
    </div>
    <details class="paper-abstract">
      The recent leaps in complexity and fluency of Large Language Models (LLMs) mean that, for the first time in human history, people can interact with computers using natural language alone. This creates monumental possibilities of automation and accessibility of computing, but also raises severe security and safety threats: When everyone can interact with LLMs, everyone can potentially break into the systems running LLMs. All it takes is creative use of language. This paper presents Hacc-Man, a game which challenges its players to "jailbreak" an LLM: subvert the LLM to output something that it is not intended to. Jailbreaking is at the intersection between creative problem solving and LLM security. The purpose of the game is threefold: 1. To heighten awareness of the risks of deploying fragile LLMs in everyday systems, 2. To heighten people's self-efficacy in interacting with LLMs, and 3. To discover the creative problem solving strategies, people deploy in this novel context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15690v1">A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 Code, data and artifacts are available: http://tinyurl.com/vrpilot-artifacts
    </div>
    <details class="paper-abstract">
      Recent work in automated program repair (APR) proposes the use of reasoning and patch validation feedback to reduce the semantic gap between the LLMs and the code under analysis. The idea has been shown to perform well for general APR, but its effectiveness in other particular contexts remains underexplored. In this work, we assess the impact of reasoning and patch validation feedback to LLMs in the context of vulnerability repair, an important and challenging task in security. To support the evaluation, we present VRpilot, an LLM-based vulnerability repair technique based on reasoning and patch validation feedback. VRpilot (1) uses a chain-of-thought prompt to reason about a vulnerability prior to generating patch candidates and (2) iteratively refines prompts according to the output of external tools (e.g., compiler, code sanitizers, test suite, etc.) on previously-generated patches. To evaluate performance, we compare VRpilot against the state-of-the-art vulnerability repair techniques for C and Java using public datasets from the literature. Our results show that VRpilot generates, on average, 14% and 7.6% more correct patches than the baseline techniques on C and Java, respectively. We show, through an ablation study, that reasoning and patch validation feedback are critical. We report several lessons from this study and potential directions for advancing LLM-empowered vulnerability repair
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15842v1">Model Cascading for Code: Reducing Inference Costs with Model Cascading for LLM Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-24
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has led to significant advancements in code completion tasks. While larger models have higher accuracy, they also cost much more to run. Meanwhile, model cascading has been proven effective to conserve computational resources while enhancing accuracy in LLMs on natural language generation tasks. It generates output with the smallest model in a set, and only queries the larger models when it fails to meet predefined quality criteria. However, this strategy has not been used in code completion tasks, primarily because assessing the quality of code completions differs substantially from assessing natural language, where the former relies heavily on the functional correctness. To address this, we propose letting each model generate and execute a set of test cases for their solutions, and use the test results as the cascading threshold. We show that our model cascading strategy reduces computational costs while increases accuracy compared to generating the output with a single model. We also introduce a heuristics to determine the optimal combination of the number of solutions, test cases, and test lines each model should generate, based on the budget. Compared to speculative decoding, our method works on black-box models, having the same level of cost-accuracy trade-off, yet providing much more choices based on the server's budget. Ours is the first work to optimize cost-accuracy trade-off for LLM code generation with model cascading.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12741v2">MuLan: Multimodal-LLM Agent for Progressive and Interactive Multi-Object Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 Added the application to human-agent interaction; added discussion with concurrent work
    </div>
    <details class="paper-abstract">
      Existing text-to-image models still struggle to generate images of multiple objects, especially in handling their spatial positions, relative sizes, overlapping, and attribute bindings. To efficiently address these challenges, we develop a training-free Multimodal-LLM agent (MuLan), as a human painter, that can progressively generate multi-object with intricate planning and feedback control. MuLan harnesses a large language model (LLM) to decompose a prompt to a sequence of sub-tasks, each generating only one object by stable diffusion, conditioned on previously generated objects. Unlike existing LLM-grounded methods, MuLan only produces a high-level plan at the beginning while the exact size and location of each object are determined upon each sub-task by an LLM and attention guidance. Moreover, MuLan adopts a vision-language model (VLM) to provide feedback to the image generated in each sub-task and control the diffusion model to re-generate the image if it violates the original prompt. Hence, each model in every step of MuLan only needs to address an easy sub-task it is specialized for. The multi-step process also allows human users to monitor the generation process and make preferred changes at any intermediate step via text prompts, thereby improving the human-AI collaboration experience. We collect 200 prompts containing multi-objects with spatial relationships and attribute bindings from different benchmarks to evaluate MuLan. The results demonstrate the superiority of MuLan in generating multiple objects over baselines and its creativity when collaborating with human users. The code is available at https://github.com/measure-infinity/mulan-code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15646v1">LLM-based Robot Task Planning with Exceptional Handling for General Purpose Service Robots</a></div>
    <div class="paper-meta">
      📅 2024-05-24
    </div>
    <details class="paper-abstract">
      The development of a general purpose service robot for daily life necessitates the robot's ability to deploy a myriad of fundamental behaviors judiciously. Recent advancements in training Large Language Models (LLMs) can be used to generate action sequences directly, given an instruction in natural language with no additional domain information. However, while the outputs of LLMs are semantically correct, the generated task plans may not accurately map to acceptable actions and might encompass various linguistic ambiguities. LLM hallucinations pose another challenge for robot task planning, which results in content that is inconsistent with real-world facts or user inputs. In this paper, we propose a task planning method based on a constrained LLM prompt scheme, which can generate an executable action sequence from a command. An exceptional handling module is further proposed to deal with LLM hallucinations problem. This module can ensure the LLM-generated results are admissible in the current environment. We evaluate our method on the commands generated by the RoboCup@Home Command Generator, observing that the robot demonstrates exceptional performance in both comprehending instructions and executing tasks.
    </details>
</div>
