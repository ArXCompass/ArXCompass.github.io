# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v3">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16781v2">Evaluating Robustness of LLMs in Question Answering on Multilingual Noisy OCR Data</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted at CIKM 2025
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors - imperfect extraction of text, including character insertion, deletion, and substitution can significantly impact downstream tasks like question-answering (QA). In this work, we conduct a comprehensive analysis of how OCR-induced noise affects the performance of Multilingual QA Systems. To support this analysis, we introduce a multilingual QA dataset MultiOCR-QA, comprising 50K question-answer pairs across three languages, English, French, and German. The dataset is curated from OCR-ed historical documents, which include different levels and types of OCR noise. We then evaluate how different state-of-the-art Large Language models (LLMs) perform under different error conditions, focusing on three major OCR error types. Our findings show that QA systems are highly prone to OCR-induced errors and perform poorly on noisy OCR text. By comparing model performance on clean versus noisy texts, we provide insights into the limitations of current approaches and emphasize the need for more noise-resilient QA systems in historical digitization contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v4">The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce unprecedented security vulnerabilities that extend beyond traditional content generation attacks to system-level compromise. This paper presents a comprehensive evaluation of the security of LLMs used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving complete computer takeover. We focus on how different attack surfaces and trust boundaries - Direct Prompt Injection, RAG Backdoor, and Inter Agent Trust - can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs (including GPT-4, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection and 83.3% are vulnerable to the more stealth and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed a critical security flaw: LLMs which successfully resist direct injection or RAG backdoor will execute identical payloads when requested by peer agents. Our findings show that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks and that every model exhibits context-dependent security behaviors that create exploitable blind spots. Our results also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11773v3">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents-virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v2">LLMs Have a Heart of Stone: Demystifying the Soft Thinking Ability of Large Reasoning Models</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 10 pages, 7 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04530v1">StyliTruth : Unlocking Stylized yet Truthful LLM Generation via Disentangled Steering</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Generating stylized large language model (LLM) responses via representation editing is a promising way for fine-grained output control. However, there exists an inherent trade-off: imposing a distinctive style often degrades truthfulness. Existing representation editing methods, by naively injecting style signals, overlook this collateral impact and frequently contaminate the model's core truthfulness representations, resulting in reduced answer correctness. We term this phenomenon stylization-induced truthfulness collapse. We attribute this issue to latent coupling between style and truth directions in certain key attention heads, and propose StyliTruth, a mechanism that preserves stylization while keeping truthfulness intact. StyliTruth separates the style-relevant and truth-relevant subspaces in the model's representation space via an orthogonal deflation process. This decomposition enables independent control of style and truth in their own subspaces, minimizing interference. By designing adaptive, token-level steering vectors within each subspace, we dynamically and precisely control the generation process to maintain both stylistic fidelity and truthfulness. We validate our method on multiple styles and languages. Extensive experiments and analyses show that StyliTruth significantly reduces stylization-induced truthfulness collapse and outperforms existing inference-time intervention methods in balancing style adherence with truthfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22716v2">From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: https://github.com/probe2/TIRESRAG-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04451v1">Automatic LLM Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Red teaming is critical for identifying vulnerabilities and building trust in current LLMs. However, current automated methods for Large Language Models (LLMs) rely on brittle prompt templates or single-turn attacks, failing to capture the complex, interactive nature of real-world adversarial dialogues. We propose a novel paradigm: training an AI to strategically `break' another AI. By formalizing red teaming as a Markov Decision Process (MDP) and employing a hierarchical Reinforcement Learning (RL) framework, we effectively address the inherent sparse reward and long-horizon challenges. Our generative agent learns coherent, multi-turn attack strategies through a fine-grained, token-level harm reward, enabling it to uncover subtle vulnerabilities missed by existing baselines. This approach sets a new state-of-the-art, fundamentally reframing LLM red teaming as a dynamic, trajectory-based process (rather than a one-step test) essential for robust AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v4">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04440v1">StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 24 pages, 17 figures, under review
    </div>
    <details class="paper-abstract">
      Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04428v1">\textsc{SimInstruct}: A Responsible Tool for Collecting Scaffolding Dialogues Between Experts and LLM-Simulated Novices</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      High-quality, multi-turn instructional dialogues between novices and experts are essential for developing AI systems that support teaching, learning, and decision-making. These dialogues often involve scaffolding -- the process by which an expert supports a novice's thinking through questions, feedback, and step-by-step guidance. However, such data are scarce due to privacy concerns in recording and the vulnerability inherent in help-seeking. We present SimInstruct, a scalable, expert-in-the-loop tool for collecting scaffolding dialogues. Using teaching development coaching as an example domain, SimInstruct simulates novice instructors via LLMs, varying their teaching challenges and LLM's persona traits, while human experts provide multi-turn feedback, reasoning, and instructional support. This design enables the creation of realistic, pedagogically rich dialogues without requiring real novice participants. Our results reveal that persona traits, such as extroversion and introversion, meaningfully influence how experts engage. Compared to real mentoring recordings, SimInstruct dialogues demonstrate comparable pedagogical relevance and cognitive depth. Experts also reported the process as engaging and reflective, improving both data quality and their own professional insight. We further fine-tuned a LLaMA model to be an expert model using the augmented dataset, which outperformed GPT-4o in instructional quality. Our analysis highlights GPT-4o's limitations in weak reflective questioning, overuse of generic praise, a condescending tone, and a tendency to overwhelm novices with excessive suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04412v1">Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Frontier LLMs only recently enabled serviceable, autonomous web agents. At that, a model poses as an instantaneous domain model backend. Ought to suggest interaction, it is consulted with a web-based task and respective application state. The key problem lies in application state serialisation $\unicode{x2013}$ referred to as snapshot. State-of-the-art web agents are premised on grounded GUI snapshots, i.e., screenshots enhanced with visual cues. Not least to resemble human perception, but for images representing relatively cheap means of model input. LLM vision still lag behind code interpretation capabilities. DOM snapshots, which structurally resemble HTML, impose a desired alternative. Vast model input token size, however, disables reliable implementation with web agents to date. We propose D2Snap, a first-of-its-kind DOM downsampling algorithm. Based on a GPT-4o backend, we evaluate D2Snap on tasks sampled from the Online-Mind2Web dataset. The success rate of D2Snap-downsampled DOM snapshots (67%) matches a grounded GUI snapshot baseline (65%) $\unicode{x2013}$ within the same input token order of magnitude (1e3). Our best evaluated configurations $\unicode{x2013}$ one token order above, but within the model's context window $\unicode{x2013}$ outperform this baseline by 8%. Our evaluation, moreover, yields that DOM-inherent hierarchy embodies a strong UI feature for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04405v1">FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate exceptional performance but entail significant memory and computational costs, restricting their practical deployment. While existing INT4/INT8 quantization reduces these costs, they often degrade accuracy or lack optimal efficiency. INT6 quantization offers a superior trade-off between model accuracy and inference efficiency, but lacks hardware support in modern GPUs, forcing emulation via higher-precision arithmetic units that limit acceleration. In this paper, we propose FlexQ, a novel post-training INT6 quantization framework combining algorithmic innovation with system-level optimizations. FlexQ employs uniform 6-bit weight quantization across all layers, with adaptive retention of 8-bit activations in layers identified through layer-wise sensitivity analysis. To maximize hardware efficiency, we develop a specialized high-performance GPU kernel supporting matrix multiplication for W6A6 and W6A8 representations via Binary Tensor Core (BTC) equivalents, effectively bypassing the lack of native INT6 tensor cores. Evaluations on LLaMA models show FlexQ maintains near-FP16 accuracy, with perplexity increases of no more than 0.05. The proposed kernel achieves an average 1.39$\times$ speedup over ABQ-LLM on LLaMA-2-70B linear layers. End-to-end, FlexQ delivers 1.33$\times$ inference acceleration and 1.21$\times$ memory savings over SmoothQuant. Code is released at https://github.com/FlyFoxPlayer/FlexQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04401v1">Why are LLMs' abilities emergent?</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) in generative tasks has raised fundamental questions about the nature of their acquired capabilities, which often appear to emerge unexpectedly without explicit training. This paper examines the emergent properties of Deep Neural Networks (DNNs) through both theoretical analysis and empirical observation, addressing the epistemological challenge of "creation without understanding" that characterises contemporary AI development. We explore how the neural approach's reliance on nonlinear, stochastic processes fundamentally differs from symbolic computational paradigms, creating systems whose macro-level behaviours cannot be analytically derived from micro-level neuron activities. Through analysis of scaling laws, grokking phenomena, and phase transitions in model capabilities, I demonstrate that emergent abilities arise from the complex dynamics of highly sensitive nonlinear systems rather than simply from parameter scaling alone. My investigation reveals that current debates over metrics, pre-training loss thresholds, and in-context learning miss the fundamental ontological nature of emergence in DNNs. I argue that these systems exhibit genuine emergent properties analogous to those found in other complex natural phenomena, where systemic capabilities emerge from cooperative interactions among simple components without being reducible to their individual behaviours. The paper concludes that understanding LLM capabilities requires recognising DNNs as a new domain of complex dynamical systems governed by universal principles of emergence, similar to those operating in physics, chemistry, and biology. This perspective shifts the focus from purely phenomenological definitions of emergence to understanding the internal dynamic transformations that enable these systems to acquire capabilities that transcend their individual components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03329v2">Industrial LLM-based Code Optimization under Regulation: A Mixture-of-Agents Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Submitted to ASE'25 Industry Showcase
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) for code optimization have enabled industrial platforms to automate software performance engineering at unprecedented scale and speed. Yet, organizations in regulated industries face strict constraints on which LLMs they can use - many cannot utilize commercial models due to data privacy regulations and compliance requirements, creating a significant challenge for achieving high-quality code optimization while maintaining cost-effectiveness. We address this by implementing a Mixture-of-Agents (MoA) approach that directly synthesizes code from multiple specialized LLMs, comparing it against TurinTech AI's vanilla Genetic Algorithm (GA)-based ensemble system and individual LLM optimizers using real-world industrial codebases. Our key contributions include: (1) First MoA application to industrial code optimization using real-world codebases; (2) Empirical evidence that MoA excels with open-source models, achieving 14.3% to 22.2% cost savings and 28.6% to 32.2% faster optimization times for regulated environments; (3) Deployment guidelines demonstrating GA's advantage with commercial models while both ensembles outperform individual LLMs; and (4) Real-world validation across 50 code snippets and seven LLM combinations, generating over 8,700 variants, addresses gaps in industrial LLM ensemble evaluation. This provides actionable guidance for organizations balancing regulatory compliance with optimization performance in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04353v1">LUST: A Multi-Modal Framework with Hierarchical LLM-based Scoring for Learned Thematic Significance Tracking in Multimedia Content</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 5 pages and 4 figures
    </div>
    <details class="paper-abstract">
      This paper introduces the Learned User Significance Tracker (LUST), a framework designed to analyze video content and quantify the thematic relevance of its segments in relation to a user-provided textual description of significance. LUST leverages a multi-modal analytical pipeline, integrating visual cues from video frames with textual information extracted via Automatic Speech Recognition (ASR) from the audio track. The core innovation lies in a hierarchical, two-stage relevance scoring mechanism employing Large Language Models (LLMs). An initial "direct relevance" score, $S_{d,i}$, assesses individual segments based on immediate visual and auditory content against the theme. This is followed by a "contextual relevance" score, $S_{c,i}$, that refines the assessment by incorporating the temporal progression of preceding thematic scores, allowing the model to understand evolving narratives. The LUST framework aims to provide a nuanced, temporally-aware measure of user-defined significance, outputting an annotated video with visualized relevance scores and comprehensive analytical logs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12286v3">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce two diagnostic tasks: file path identification from issue descriptions alone and ground truth function reproduction with only the current file context and issue description to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. Similar patterns are also observed for the function reproduction task, where the verbatim similarity is much higher on SWE-Bench Verified than on other similar coding benchmarks (up to 35% consecutive 5-gram accuracy on SWE-Bench Verified and Full, but only up to 18% for tasks in other benchmarks). These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13287v5">PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Selecting a sample generation scheme from multiple prompt-based generative models, including large language models (LLMs) and prompt-guided image and video generation models, is typically addressed by choosing the model that maximizes an averaged evaluation score. However, this score-based selection overlooks the possibility that different models achieve the best generation performance for different types of text prompts. An online identification of the best generation model for various input prompts can reduce the costs associated with querying sub-optimal models. In this work, we explore the possibility of varying rankings of text-based generative models for different text prompts and propose an online learning framework to predict the best data generation model for a given input prompt. The proposed PAK-UCB algorithm addresses a contextual bandit (CB) setting with shared context variables across the arms, utilizing the generated data to update kernel-based functions that predict the score of each model available for unseen text prompts. Additionally, we leverage random Fourier features (RFF) to accelerate the online learning process of PAK-UCB. Our numerical experiments on real and simulated text-to-image and image-to-text generative models show that RFF-UCB performs successfully in identifying the best generation model across different sample types. The code is available at: github.com/yannxiaoyanhu/dgm-online-select.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04279v1">Mockingbird: How does LLM perform in general machine learning tasks?</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now being used with increasing frequency as chat bots, tasked with the summarizing information or generating text and code in accordance with user instructions. The rapid increase in reasoning capabilities and inference speed of LLMs has revealed their remarkable potential for applications extending beyond the domain of chat bots to general machine learning tasks. This work is conducted out of the curiosity about such potential. In this work, we propose a framework Mockingbird to adapt LLMs to general machine learning tasks and evaluate its performance and scalability on several general machine learning tasks. The core concept of this framework is instructing LLMs to role-play functions and reflect on its mistakes to improve itself. Our evaluation and analysis result shows that LLM-driven machine learning methods, such as Mockingbird, can achieve acceptable results on common machine learning tasks; however, solely reflecting on its own currently cannot outperform the effect of domain-specific documents and feedback from human experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14448v2">How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04257v1">KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      Key-Value (KV) cache quantization has become a widely adopted optimization technique for efficient large language models (LLMs) inference by reducing KV cache memory usage and mitigating memory-bound constraints. Recent studies have emphasized the importance of preserving the original precision of KVs for the first few tokens to ensure the protection of attention sinks. While this approach has proven effective in mitigating performance degradation, its underlying principles remain insufficiently understood. Moreover, it fails to address the recent discovery that attention sinks can emerge beyond the initial token positions. In this work, we elucidate the underlying mechanisms of attention sinks during inference by examining their role in the cross-layer evolution of extreme activation outliers. Additionally, we provide a comprehensive analysis of the interplay between attention sinks and KV cache quantization. Based on our enhanced understanding, we introduce \textit{\textbf{KVSink}}, a plug-and-play method that effectively predicts sink tokens with negligible overhead, enabling more thorough preservation. Extensive experiments demonstrate that KVSink outperforms the existing Preserve-First-N (PFN) strategy, offering more effective preservation of attention sinks during KV cache quantization. Moreover, when applied to the well-established KVQuant method, KVSink further improves perplexity (PPL) and reduces reliance on 16-bit numerical outliers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04231v1">Empowering Time Series Forecasting with LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04206v1">ViLLA-MMBench: A Unified Benchmark Suite for LLM-Augmented Multimodal Movie Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 17 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Recommending long-form video content demands joint modeling of visual, audio, and textual modalities, yet most benchmarks address only raw features or narrow fusion. We present ViLLA-MMBench, a reproducible, extensible benchmark for LLM-augmented multimodal movie recommendation. Built on MovieLens and MMTF-14K, it aligns dense item embeddings from three modalities: audio (block-level, i-vector), visual (CNN, AVF), and text. Missing or sparse metadata is automatically enriched using state-of-the-art LLMs (e.g., OpenAI Ada), generating high-quality synopses for thousands of movies. All text (raw or augmented) is embedded with configurable encoders (Ada, LLaMA-2, Sentence-T5), producing multiple ready-to-use sets. The pipeline supports interchangeable early-, mid-, and late-fusion (concatenation, PCA, CCA, rank-aggregation) and multiple backbones (MF, VAECF, VBPR, AMR, VMF) for ablation. Experiments are fully declarative via a single YAML file. Evaluation spans accuracy (Recall, nDCG) and beyond-accuracy metrics: cold-start rate, coverage, novelty, diversity, fairness. Results show LLM-based augmentation and strong text embeddings boost cold-start and coverage, especially when fused with audio-visual features. Systematic benchmarking reveals universal versus backbone- or metric-specific combinations. Open-source code, embeddings, and configs enable reproducible, fair multimodal RS research and advance principled generative AI integration in large-scale recommendation. Code: https://recsys-lab.github.io/ViLLA-MMBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12342v2">CRAB: A Benchmark for Evaluating Curation of Retrieval-Augmented LLMs in Biomedicine</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Recent development in Retrieval-Augmented Large Language Models (LLMs) have shown great promise in biomedical applications. How ever, a critical gap persists in reliably evaluating their curation ability the process by which models select and integrate relevant references while filtering out noise. To address this, we introduce the benchmark for Curation of Retrieval-Augmented LLMs in Biomedicine (CRAB), the first multilingual benchmark tailored for evaluating the biomedical curation of retrieval-augmented LLMs, available in English, French, German and Chinese. By incorporating a novel citation-based evaluation metric, CRAB quantifies the curation performance of retrieval-augmented LLMs in biomedicine. Experimental results reveal significant discrepancies in the curation performance of mainstream LLMs, underscoring the urgent need to improve it in the domain of biomedicine. Our dataset is available at https://huggingface.co/datasets/zhm0/CRAB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04199v1">Reasoning Beyond Labels: Measuring LLM Sentiment in Low-Resource, Culturally Nuanced Contexts</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Sentiment analysis in low-resource, culturally nuanced contexts challenges conventional NLP approaches that assume fixed labels and universal affective expressions. We present a diagnostic framework that treats sentiment as a context-dependent, culturally embedded construct, and evaluate how large language models (LLMs) reason about sentiment in informal, code-mixed WhatsApp messages from Nairobi youth health groups. Using a combination of human-annotated data, sentiment-flipped counterfactuals, and rubric-based explanation evaluation, we probe LLM interpretability, robustness, and alignment with human reasoning. Framing our evaluation through a social-science measurement lens, we operationalize and interrogate LLMs outputs as an instrument for measuring the abstract concept of sentiment. Our findings reveal significant variation in model reasoning quality, with top-tier LLMs demonstrating interpretive stability, while open models often falter under ambiguity or sentiment shifts. This work highlights the need for culturally sensitive, reasoning-aware AI evaluation in complex, real-world communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03958v2">A Comparative Study of Specialized LLMs as Dense Retrievers</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted by CCIR25 and published by Springer LNCS or LNAI
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02096v2">Evaluating User Experience in Conversational Recommender Systems: A Systematic Review Across Classical and LLM-Powered Approaches</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted at OzCHI 2025. 23 pages, 1 figure, 5 tables
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRSs) are receiving growing research attention across domains, yet their user experience (UX) evaluation remains limited. Existing reviews largely overlook empirical UX studies, particularly in adaptive and large language model (LLM)-based CRSs. To address this gap, we conducted a systematic review following PRISMA guidelines, synthesising 23 empirical studies published between 2017 and 2025. We analysed how UX has been conceptualised, measured, and shaped by domain, adaptivity, and LLM. Our findings reveal persistent limitations: post hoc surveys dominate, turn-level affective UX constructs are rarely assessed, and adaptive behaviours are seldom linked to UX outcomes. LLM-based CRSs introduce further challenges, including epistemic opacity and verbosity, yet evaluations infrequently address these issues. We contribute a structured synthesis of UX metrics, a comparative analysis of adaptive and nonadaptive systems, and a forward-looking agenda for LLM-aware UX evaluation. These findings support the development of more transparent, engaging, and user-centred CRS evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15395v2">Parse Trees Guided LLM Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 IEEE TPAMI major revision submitted
    </div>
    <details class="paper-abstract">
      Offering rich contexts to Large Language Models (LLMs) has shown to boost the performance in various tasks, but the resulting longer prompt would increase the computational cost and might exceed the input limit of LLMs. Recently, some prompt compression methods have been suggested to shorten the length of prompts by using language models to generate shorter prompts or by developing computational models to select important parts of original prompt. The generative compression methods would suffer from issues like hallucination, while the selective compression methods have not involved linguistic rules and overlook the global structure of prompt. To this end, we propose a novel selective compression method called PartPrompt. It first obtains a parse tree for each sentence based on linguistic rules, and calculates local information entropy for each node in a parse tree. These local parse trees are then organized into a global tree according to the hierarchical structure such as the dependency of sentences, paragraphs, and sections. After that, the root-ward propagation and leaf-ward propagation are proposed to adjust node values over the global tree. Finally, a recursive algorithm is developed to prune the global tree based on the adjusted node values. The experiments show that PartPrompt receives the state-of-the-art performance across various datasets, metrics, compression ratios, and target LLMs for inference. The in-depth ablation studies confirm the effectiveness of designs in PartPrompt, and other additional experiments also demonstrate its superiority in terms of the coherence of compressed prompts and in the extreme long prompt scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16888v2">CAIN: Hijacking LLM-Humans Conversations via Malicious System Prompts</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04117v1">Unveiling Over-Memorization in Finetuning LLMs for Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      The pretrained large language models (LLMs) are finetuned with labeled data for better instruction following ability and alignment with human values. In this paper, we study the learning dynamics of LLM finetuning on reasoning tasks and reveal the uncovered over-memorization phenomenon during a specific stage of LLM finetuning. At this stage, the LLMs have excessively memorized training data and exhibit high test perplexity while maintaining good test accuracy. We investigate the conditions that lead to LLM over-memorization and find that training epochs and large learning rates contribute to this issue. Although models with over-memorization demonstrate comparable test accuracy to normal models, they suffer from reduced robustness, poor out-of-distribution generalization, and decreased generation diversity. Our experiments unveil the over-memorization to be broadly applicable across different tasks, models, and finetuning methods. Our research highlights that overparameterized, extensively finetuned LLMs exhibit unique learning dynamics distinct from traditional machine learning models. Based on our observations of over-memorization, we provide recommendations on checkpoint and learning rate selection during finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06043v2">CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to ACL 2025 (Findings), camera-ready version
    </div>
    <details class="paper-abstract">
      Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23989v3">Rubric Is All You Need: Enhancing LLM-based Code Evaluation With Question-Specific Rubrics</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted in ICER 2025
    </div>
    <details class="paper-abstract">
      Since the emergence of Large Language Models (LLMs) popularized by the release of GPT-3 and ChatGPT, LLMs have shown remarkable promise in programming-related tasks. While code generation using LLMs has become a popular field of research, code evaluation using LLMs remains under-explored. In this paper, we focus on LLM-based code evaluation and attempt to fill in the existing gaps. We propose multi-agentic novel approaches using \emph{question-specific rubrics} tailored to the problem statement, arguing that these perform better for logical assessment than the existing approaches that use \emph{question-agnostic rubrics}. To address the lack of suitable evaluation datasets, we introduce two datasets: a Data Structures and Algorithms dataset containing 150 student submissions from a popular Data Structures and Algorithms practice website, and an Object Oriented Programming dataset comprising 80 student submissions from undergraduate computer science courses. In addition to using standard metrics (Spearman Correlation, Cohen's Kappa), we additionally propose a new metric called as Leniency, which quantifies evaluation strictness relative to expert assessment. Our comprehensive analysis demonstrates that \emph{question-specific rubrics} significantly enhance logical assessment of code in educational settings, providing better feedback aligned with instructional goals beyond mere syntactic correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04096v1">Efficient Scaling for LLM-based ASR</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted by ASRU 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based automatic speech recognition (ASR) achieves strong performance but often incurs high computational costs. This work investigates how to obtain the best LLM-ASR performance efficiently. Through comprehensive and controlled experiments, we find that pretraining the speech encoder before integrating it with the LLM leads to significantly better scaling efficiency than the standard practice of joint post-training of LLM-ASR. Based on this insight, we propose a new multi-stage LLM-ASR training strategy, EFIN: Encoder First Integration. Among all training strategies evaluated, EFIN consistently delivers better performance (relative to 21.1% CERR) with significantly lower computation budgets (49.9% FLOPs). Furthermore, we derive a scaling law that approximates ASR error rates as a computation function, providing practical guidance for LLM-ASR scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04073v1">Efficient Strategy for Improving Large Language Model (LLM) Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Based on master's thesis in Systems and Computer Engineering, Universidad Nacional de Colombia (2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become a milestone in the field of artificial intelligence and natural language processing. However, their large-scale deployment remains constrained by the need for significant computational resources. This work proposes starting from a base model to explore and combine data processing and careful data selection techniques, training strategies, and architectural adjustments to improve the efficiency of LLMs in resource-constrained environments and within a delimited knowledge base. The methodological approach included defining criteria for building reliable datasets, conducting controlled experiments with different configurations, and systematically evaluating the resulting variants in terms of capability, versatility, response time, and safety. Finally, comparative tests were conducted to measure the performance of the developed variants and to validate the effectiveness of the proposed strategies. This work is based on the master's thesis in Systems and Computer Engineering titled "Efficient Strategy for Improving the Capabilities of Large Language Models (LLMs)".
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02381v2">Beyond Manually Designed Pruning Policies with Second-Level Performance Prediction: A Pruning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Non-uniform structured network pruning methods can effectively reduce Large Language Model (LLM) size by eliminating redundant channels or layers, offering lower performance degradation than uniform strategies. However, existing non-uniform methods rely heavily on manually designed pruning policies (e.g., layer importance and scaling factors), and therefore cannot efficiently adapt to scenarios with dynamic pruning ratio requirements. Additionly, a critical bottleneck -- the time-consuming evaluation of pruning policies -- further limits the feasibility of iteratively and dynamically finding optimal pruning policies. To address these limitations, we propose PPF (Predictive Pruning Framework), a novel pruning framework for LLMs that eliminates manual design dependencies via second-level performance prediction. PPF not only supports real-time pruning decisions under dynamic pruning ratios but is also applicable to static pruning scenarios. It employs an agent for producing adaptive and real-time pruning actions, while a lightweight performance predictor that can evaluate a pruning policy in seconds, significantly speeding up the iterative optimization process. Experiments on Llama2-7B and Llama3-8B show that PPF can generate dynamic/static pruning policies and it reduces perplexity by up to 33.4% (dynamic pruning) and 84.78% (static pruning) over existing methods, outperforming manually designed pruning policies. The performance predictor achieves second-level performance prediction with high accuracy (prediction error < 0.0011). It reduces the mean evaluation latency from minute-level (1 minute and 38.02 seconds of test-set evaluation methods) to second-level (1.52 seconds), achieving over 64 times speedup. Our code will be available at https://github.com/Ma-zx/PPF .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22548v2">Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Long chain-of-thought (CoT) reasoning has shown great promise in enhancing the emotion understanding performance of large language models (LLMs). However, current fixed-length CoT methods struggle to balance reasoning depth and efficiency. Simple tasks (e.g., sentiment classification) are over-reasoned, while complex tasks (e.g., sarcasm understanding) lack depth. To fill this gap, we present Emotion-o1, an adaptive CoT framework that dynamically adjusts reasoning length based on emotion-task complexity. Emotion-o1 is trained by distilling adaptive CoT patterns from a reasoning-oriented LLM, followed by supervised fine-tuning and reinforcement learning with a four-part reward targeting accuracy, brevity, structure, and redundancy. Experimental results on four emotion tasks highlight: (1) Emotion-o1 demonstrates significant improvements over its backbone, with F1 score increases of 10%(Sentiment), 5%(Emotion), 18%(Humor), and 27%(Sarcasm). (2) In sentiment and sarcasm tasks, our 8B model demonstrates superior performance against advanced LLMs, outperforming Grok-3 by 1.1% and Claude-3.7 by 2%. (3) The framework maintains accuracy while reducing reasoning length by 83% compared to OpenAI-o1, demonstrating effective precision-efficiency optimization. Emotion-o1 effectively balances reasoning depth and efficiency for emotion understanding in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v2">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/wanghuacan/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01083v2">Tool Unlearning for Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 ICML 2025 https://clu-uml.github.io/MU-Bench-Project-Page/
    </div>
    <details class="paper-abstract">
      Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04038v1">ZARA: Zero-shot Motion Time-Series Analysis via Knowledge and Retrieval Driven LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Motion sensor time-series are central to human activity recognition (HAR), with applications in health, sports, and smart devices. However, existing methods are trained for fixed activity sets and require costly retraining when new behaviours or sensor setups appear. Recent attempts to use large language models (LLMs) for HAR, typically by converting signals into text or images, suffer from limited accuracy and lack verifiable interpretability. We propose ZARA, the first agent-based framework for zero-shot, explainable HAR directly from raw motion time-series. ZARA integrates an automatically derived pair-wise feature knowledge base that captures discriminative statistics for every activity pair, a multi-sensor retrieval module that surfaces relevant evidence, and a hierarchical agent pipeline that guides the LLM to iteratively select features, draw on this evidence, and produce both activity predictions and natural-language explanations. ZARA enables flexible and interpretable HAR without any fine-tuning or task-specific classifiers. Extensive experiments on 8 HAR benchmarks show that ZARA achieves SOTA zero-shot performance, delivering clear reasoning while exceeding the strongest baselines by 2.53x in macro F1. Ablation studies further confirm the necessity of each module, marking ZARA as a promising step toward trustworthy, plug-and-play motion time-series analysis. Our codes are available at https://github.com/zechenli03/ZARA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05758v3">APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with LLMs remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, modelagnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proofgeneration results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sublemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low budget. The repaired subproofs are recombined and reverified, iterating up to a usercontrolled maximum number of attempts. On the miniF2F benchmark, we establish a new stateoftheart accuracy of 84.9% among sub 8Bparameter models while keeping the sampling budget below one hundred. Moreover, Apollo raises the stateoftheart accuracy for GoedelProverSFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. Generalpurpose models (o3mini, o4mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compilerguided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving. The codebase is available at https://github.com/aziksh-ospanov/APOLLO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17432v3">Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 13 pages, 8 figures, Accepted by ACM MM2025, Project page: https://garygutc.github.io/UniME
    </div>
    <details class="paper-abstract">
      The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored.In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLM\'s language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short and long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21563v2">Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Recommendation systems often suffer from data sparsity caused by limited user-item interactions, which degrade their performance and amplify popularity bias in real-world scenarios. This paper proposes a novel data augmentation framework that leverages Large Language Models (LLMs) and item textual descriptions to enrich interaction data. By few-shot prompting LLMs multiple times to rerank items and aggregating the results via majority voting, we generate high-confidence synthetic user-item interactions, supported by theoretical guarantees based on the concentration of measure. To effectively leverage the augmented data in the context of a graph recommendation system, we integrate it into a graph contrastive learning framework to mitigate distributional shift and alleviate popularity bias. Extensive experiments show that our method improves accuracy and reduces popularity bias, outperforming strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03991v1">Galaxy: A Cognition-Centered Framework for Proactive, Privacy-Preserving, and Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Intelligent personal assistants (IPAs) such as Siri and Google Assistant are designed to enhance human capabilities and perform tasks on behalf of users. The emergence of LLM agents brings new opportunities for the development of IPAs. While responsive capabilities have been widely studied, proactive behaviors remain underexplored. Designing an IPA that is proactive, privacy-preserving, and capable of self-evolution remains a significant challenge. Designing such IPAs relies on the cognitive architecture of LLM agents. This work proposes Cognition Forest, a semantic structure designed to align cognitive modeling with system-level design. We unify cognitive architecture and system design into a self-reinforcing loop instead of treating them separately. Based on this principle, we present Galaxy, a framework that supports multidimensional interactions and personalized capability generation. Two cooperative agents are implemented based on Galaxy: KoRa, a cognition-enhanced generative agent that supports both responsive and proactive skills; and Kernel, a meta-cognition-based meta-agent that enables Galaxy's self-evolution and privacy preservation. Experimental results show that Galaxy outperforms multiple state-of-the-art benchmarks. Ablation studies and real-world interaction cases validate the effectiveness of Galaxy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03990v1">Are Today's LLMs Ready to Explain Well-Being Concepts?</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 9 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Well-being encompasses mental, physical, and social dimensions essential to personal growth and informed life decisions. As individuals increasingly consult Large Language Models (LLMs) to understand well-being, a key challenge emerges: Can LLMs generate explanations that are not only accurate but also tailored to diverse audiences? High-quality explanations require both factual correctness and the ability to meet the expectations of users with varying expertise. In this work, we construct a large-scale dataset comprising 43,880 explanations of 2,194 well-being concepts, generated by ten diverse LLMs. We introduce a principle-guided LLM-as-a-judge evaluation framework, employing dual judges to assess explanation quality. Furthermore, we show that fine-tuning an open-source LLM using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) can significantly enhance the quality of generated explanations. Our results reveal: (1) The proposed LLM judges align well with human evaluations; (2) explanation quality varies significantly across models, audiences, and categories; and (3) DPO- and SFT-finetuned models outperform their larger counterparts, demonstrating the effectiveness of preference-based learning for specialized explanation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04939v1">I Think, Therefore I Am Under-Qualified? A Benchmark for Evaluating Linguistic Shibboleth Detection in LLM Hiring Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      This paper introduces a comprehensive benchmark for evaluating how Large Language Models (LLMs) respond to linguistic shibboleths: subtle linguistic markers that can inadvertently reveal demographic attributes such as gender, social class, or regional background. Through carefully constructed interview simulations using 100 validated question-response pairs, we demonstrate how LLMs systematically penalize certain linguistic patterns, particularly hedging language, despite equivalent content quality. Our benchmark generates controlled linguistic variations that isolate specific phenomena while maintaining semantic equivalence, which enables the precise measurement of demographic bias in automated evaluation systems. We validate our approach along multiple linguistic dimensions, showing that hedged responses receive 25.6% lower ratings on average, and demonstrate the benchmark's effectiveness in identifying model-specific biases. This work establishes a foundational framework for detecting and measuring linguistic discrimination in AI systems, with broad applications to fairness in automated decision-making contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16086v2">Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04903v1">RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems have shown strong potential in complex reasoning and collaborative decision-making tasks. However, most existing coordination schemes rely on static or full-context routing strategies, which lead to excessive token consumption, redundant memory exposure, and limited adaptability across interaction rounds. We introduce RCR-Router, a modular and role-aware context routing framework designed to enable efficient, adaptive collaboration in multi-agent LLMs. To our knowledge, this is the first routing approach that dynamically selects semantically relevant memory subsets for each agent based on its role and task stage, while adhering to a strict token budget. A lightweight scoring policy guides memory selection, and agent outputs are iteratively integrated into a shared memory store to facilitate progressive context refinement. To better evaluate model behavior, we further propose an Answer Quality Score metric that captures LLM-generated explanations beyond standard QA accuracy. Experiments on three multi-hop QA benchmarks -- HotPotQA, MuSiQue, and 2WikiMultihop -- demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality. These results highlight the importance of structured memory routing and output-aware evaluation in advancing scalable multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13417v3">RLTHF: Targeted Human Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Presented at ICML 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04894v1">Adversarial Attacks and Defenses on Graph-aware Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated with graph-structured data for tasks like node classification, a domain traditionally dominated by Graph Neural Networks (GNNs). While this integration leverages rich relational information to improve task performance, their robustness against adversarial attacks remains unexplored. We take the first step to explore the vulnerabilities of graph-aware LLMs by leveraging existing adversarial attack methods tailored for graph-based models, including those for poisoning (training-time attacks) and evasion (test-time attacks), on two representative models, LLAGA (Chen et al. 2024) and GRAPHPROMPTER (Liu et al. 2024). Additionally, we discover a new attack surface for LLAGA where an attacker can inject malicious nodes as placeholders into the node sequence template to severely degrade its performance. Our systematic analysis reveals that certain design choices in graph encoding can enhance attack success, with specific findings that: (1) the node sequence template in LLAGA increases its vulnerability; (2) the GNN encoder used in GRAPHPROMPTER demonstrates greater robustness; and (3) both approaches remain susceptible to imperceptible feature perturbation attacks. Finally, we propose an end-to-end defense framework GALGUARD, that combines an LLM-based feature correction module to mitigate feature-level perturbations and adapted GNN defenses to protect against structural attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04450v2">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 18 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04842v1">Charts-of-Thought: Enhancing LLM Visualization Literacy Through Structured Data Extraction</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 11 pages, 8 figures. Accepted at IEEE VIS: Visualization & Visual Analytics 2025 conference, November 2-7, 2025, Vienna, Austria
    </div>
    <details class="paper-abstract">
      This paper evaluates the visualization literacy of modern Large Language Models (LLMs) and introduces a novel prompting technique called Charts-of-Thought. We tested three state-of-the-art LLMs (Claude-3.7-sonnet, GPT-4.5 preview, and Gemini-2.0-pro) on the Visualization Literacy Assessment Test (VLAT) using standard prompts and our structured approach. The Charts-of-Thought method guides LLMs through a systematic data extraction, verification, and analysis process before answering visualization questions. Our results show Claude-3.7-sonnet achieved a score of 50.17 using this method, far exceeding the human baseline of 28.82. This approach improved performance across all models, with score increases of 21.8% for GPT-4.5, 9.4% for Gemini-2.0, and 13.5% for Claude-3.7 compared to standard prompting. The performance gains were consistent across original and modified VLAT charts, with Claude correctly answering 100% of questions for several chart types that previously challenged LLMs. Our study reveals that modern multimodal LLMs can surpass human performance on visualization literacy tasks when given the proper analytical framework. These findings establish a new benchmark for LLM visualization literacy and demonstrate the importance of structured prompting strategies for complex visual interpretation tasks. Beyond improving LLM visualization literacy, Charts-of-Thought could also enhance the accessibility of visualizations, potentially benefiting individuals with visual impairments or lower visualization literacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04826v1">Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large language models require consistent behavioral patterns for safe deployment, yet their personality-like traits remain poorly understood. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25+ open-source models (1B-671B parameters) across 500,000+ responses. Using traditional (BFI-44, SD3) and novel LLM-adapted personality instruments, we systematically vary question order, paraphrasing, personas, and reasoning modes. Our findings challenge fundamental deployment assumptions: (1) Even 400B+ models exhibit substantial response variability (SD > 0.4); (2) Minor prompt reordering alone shifts personality measurements by up to 20%; (3) Interventions expected to stabilize behavior, such as chain-of-thought reasoning, detailed personas instruction, inclusion of conversation history, can paradoxically increase variability; (4) LLM-adapted instruments show equal instability to human-centric versions, confirming architectural rather than translational limitations. This persistent instability across scales and mitigation strategies suggests current LLMs lack the foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that personality-based alignment strategies may be fundamentally inadequate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04820v1">Automated File-Level Logging Generation for Machine Learning Applications using LLMs: A Case Study using GPT-4o Mini</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Logging is essential in software development, helping developers monitor system behavior and aiding in debugging applications. Given the ability of large language models (LLMs) to generate natural language and code, researchers are exploring their potential to generate log statements. However, prior work focuses on evaluating logs introduced in code functions, leaving file-level log generation underexplored -- especially in machine learning (ML) applications, where comprehensive logging can enhance reliability. In this study, we evaluate the capacity of GPT-4o mini as a case study to generate log statements for ML projects at file level. We gathered a set of 171 ML repositories containing 4,073 Python files with at least one log statement. We identified and removed the original logs from the files, prompted the LLM to generate logs for them, and evaluated both the position of the logs and log level, variables, and text quality of the generated logs compared to human-written logs. In addition, we manually analyzed a representative sample of generated logs to identify common patterns and challenges. We find that the LLM introduces logs in the same place as humans in 63.91% of cases, but at the cost of a high overlogging rate of 82.66%. Furthermore, our manual analysis reveals challenges for file-level logging, which shows overlogging at the beginning or end of a function, difficulty logging within large code blocks, and misalignment with project-specific logging conventions. While the LLM shows promise for generating logs for complete files, these limitations remain to be addressed for practical implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04795v1">Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted in the 2025 IEEE Automatic Speech Recognition and Understanding Workshop
    </div>
    <details class="paper-abstract">
      In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04787v1">Evaluating the Impact of LLM-guided Reflection on Learning Outcomes with Interactive AI-Generated Educational Podcasts</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to NCME Special Interest Group on AI in Measurement: AIME-CON 2025 conference
    </div>
    <details class="paper-abstract">
      This study examined whether embedding LLM-guided reflection prompts in an interactive AI-generated podcast improved learning and user experience compared to a version without prompts. Thirty-six undergraduates participated, and while learning outcomes were similar across conditions, reflection prompts reduced perceived attractiveness, highlighting a call for more research on reflective interactivity design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05694v1">DMFI: Dual-Modality Fine-Tuning and Inference Framework for LLM-Based Insider Threat Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Submitted to the 2025 IEEE International Conference on Data Mining (ICDM)
    </div>
    <details class="paper-abstract">
      Insider threat detection (ITD) poses a persistent and high-impact challenge in cybersecurity due to the subtle, long-term, and context-dependent nature of malicious insider behaviors. Traditional models often struggle to capture semantic intent and complex behavior dynamics, while existing LLM-based solutions face limitations in prompt adaptability and modality coverage. To bridge this gap, we propose DMFI, a dual-modality framework that integrates semantic inference with behavior-aware fine-tuning. DMFI converts raw logs into two structured views: (1) a semantic view that processes content-rich artifacts (e.g., emails, https) using instruction-formatted prompts; and (2) a behavioral abstraction, constructed via a 4W-guided (When-Where-What-Which) transformation to encode contextual action sequences. Two LoRA-enhanced LLMs are fine-tuned independently, and their outputs are fused via a lightweight MLP-based decision module. We further introduce DMFI-B, a discriminative adaptation strategy that separates normal and abnormal behavior representations, improving robustness under severe class imbalance. Experiments on CERT r4.2 and r5.2 datasets demonstrate that DMFI outperforms state-of-the-art methods in detection accuracy. Our approach combines the semantic reasoning power of LLMs with structured behavior modeling, offering a scalable and effective solution for real-world insider threat detection. Our work demonstrates the effectiveness of combining LLM reasoning with structured behavioral modeling, offering a scalable and deployable solution for modern insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05687v1">Risk Analysis Techniques for Governed LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Organisations are starting to adopt LLM-based AI agents, with their deployments naturally evolving from single agents towards interconnected, multi-agent networks. Yet a collection of safe agents does not guarantee a safe collection of agents, as interactions between agents over time create emergent behaviours and induce novel failure modes. This means multi-agent systems require a fundamentally different risk analysis approach than that used for a single agent. This report addresses the early stages of risk identification and analysis for multi-agent AI systems operating within governed environments where organisations control their agent configurations and deployment. In this setting, we examine six critical failure modes: cascading reliability failures, inter-agent communication failures, monoculture collapse, conformity bias, deficient theory of mind, and mixed motive dynamics. For each, we provide a toolkit for practitioners to extend or integrate into their existing frameworks to assess these failure modes within their organisational contexts. Given fundamental limitations in current LLM behavioural understanding, our approach centres on analysis validity, and advocates for progressively increasing validity through staged testing across stages of abstraction and deployment that gradually increases exposure to potential negative impacts, while collecting convergent evidence through simulation, observational analysis, benchmarking, and red teaming. This methodology establishes the groundwork for robust organisational risk management as these LLM-based multi-agent systems are deployed and operated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03153v1">Estimating Worst-Case Frontier Risks of Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      In this paper, we study the worst-case frontier risks of releasing gpt-oss. We introduce malicious fine-tuning (MFT), where we attempt to elicit maximum capabilities by fine-tuning gpt-oss to be as capable as possible in two domains: biology and cybersecurity. To maximize biological risk (biorisk), we curate tasks related to threat creation and train gpt-oss in an RL environment with web browsing. To maximize cybersecurity risk, we train gpt-oss in an agentic coding environment to solve capture-the-flag (CTF) challenges. We compare these MFT models against open- and closed-weight LLMs on frontier risk evaluations. Compared to frontier closed-weight models, MFT gpt-oss underperforms OpenAI o3, a model that is below Preparedness High capability level for biorisk and cybersecurity. Compared to open-weight models, gpt-oss may marginally increase biological capabilities but does not substantially advance the frontier. Taken together, these results contributed to our decision to release the model, and we hope that our MFT approach can serve as useful guidance for estimating harm from future open-weight releases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03148v1">Frontier: Simulating the Next Generation of LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is growing increasingly complex with the rise of Mixture-of-Experts (MoE) models and disaggregated architectures that decouple components like prefill/decode (PD) or attention/FFN (AF) for heterogeneous scaling. Existing simulators, architected for co-located, dense models, are unable to capture the intricate system dynamics of these emerging paradigms. We present Frontier, a high-fidelity simulator designed from the ground up for this new landscape. Frontier introduces a unified framework to model both co-located and disaggregated systems, providing native support for MoE inference with expert parallelism (EP). It enables the simulation of complex workflows like cross-cluster expert routing and advanced pipelining strategies for latency hiding. To ensure fidelity and usability, Frontier incorporates refined operator models for improved accuracy. Frontier empowers the community to design and optimize the future of LLM inference at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03125v1">Attack the Messages, Not the Agents: A Multi-round Adaptive Stealthy Tampering Framework for LLM-MAS</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large language model-based multi-agent systems (LLM-MAS) effectively accomplish complex and dynamic tasks through inter-agent communication, but this reliance introduces substantial safety vulnerabilities. Existing attack methods targeting LLM-MAS either compromise agent internals or rely on direct and overt persuasion, which limit their effectiveness, adaptability, and stealthiness. In this paper, we propose MAST, a Multi-round Adaptive Stealthy Tampering framework designed to exploit communication vulnerabilities within the system. MAST integrates Monte Carlo Tree Search with Direct Preference Optimization to train an attack policy model that adaptively generates effective multi-round tampering strategies. Furthermore, to preserve stealthiness, we impose dual semantic and embedding similarity constraints during the tampering process. Comprehensive experiments across diverse tasks, communication architectures, and LLMs demonstrate that MAST consistently achieves high attack success rates while significantly enhancing stealthiness compared to baselines. These findings highlight the effectiveness, stealthiness, and adaptability of MAST, underscoring the need for robust communication safeguards in LLM-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03097v1">VFLAIR-LLM: A Comprehensive Framework and Benchmark for Split Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 12 pages, 10 figures, published in KDD2025
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Models (LLMs), LLM applications have expanded into a growing number of fields. However, users with data privacy concerns face limitations in directly utilizing LLM APIs, while private deployments incur significant computational demands. This creates a substantial challenge in achieving secure LLM adaptation under constrained local resources. To address this issue, collaborative learning methods, such as Split Learning (SL), offer a resource-efficient and privacy-preserving solution for adapting LLMs to private domains. In this study, we introduce VFLAIR-LLM (available at https://github.com/FLAIR-THU/VFLAIR-LLM), an extensible and lightweight split learning framework for LLMs, enabling privacy-preserving LLM inference and fine-tuning in resource-constrained environments. Our library provides two LLM partition settings, supporting three task types and 18 datasets. In addition, we provide standard modules for implementing and evaluating attacks and defenses. We benchmark 5 attacks and 9 defenses under various Split Learning for LLM(SL-LLM) settings, offering concrete insights and recommendations on the choice of model partition configurations, defense strategies, and relevant hyperparameters for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03094v1">Augmenting Continual Learning of Diseases with LLM-Generated Visual Concepts</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Continual learning is essential for medical image classification systems to adapt to dynamically evolving clinical environments. The integration of multimodal information can significantly enhance continual learning of image classes. However, while existing approaches do utilize textual modality information, they solely rely on simplistic templates with a class name, thereby neglecting richer semantic information. To address these limitations, we propose a novel framework that harnesses visual concepts generated by large language models (LLMs) as discriminative semantic guidance. Our method dynamically constructs a visual concept pool with a similarity-based filtering mechanism to prevent redundancy. Then, to integrate the concepts into the continual learning process, we employ a cross-modal image-concept attention module, coupled with an attention loss. Through attention, the module can leverage the semantic knowledge from relevant visual concepts and produce class-representative fused features for classification. Experiments on medical and natural image datasets show our method achieves state-of-the-art performance, demonstrating the effectiveness and superiority of our method. We will release the code publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03092v1">Toward Verifiable Misinformation Detection: A Multi-Tool LLM Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs), the detection of misinformation has become increasingly important and complex. This research proposes an innovative verifiable misinformation detection LLM agent that goes beyond traditional true/false binary judgments. The agent actively verifies claims through dynamic interaction with diverse web sources, assesses information source credibility, synthesizes evidence, and provides a complete verifiable reasoning process. Our designed agent architecture includes three core tools: precise web search tool, source credibility assessment tool and numerical claim verification tool. These tools enable the agent to execute multi-step verification strategies, maintain evidence logs, and form comprehensive assessment conclusions. We evaluate using standard misinformation datasets such as FakeNewsNet, comparing with traditional machine learning models and LLMs. Evaluation metrics include standard classification metrics, quality assessment of reasoning processes, and robustness testing against rewritten content. Experimental results show that our agent outperforms baseline methods in misinformation detection accuracy, reasoning transparency, and resistance to information rewriting, providing a new paradigm for trustworthy AI-assisted fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03082v1">EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Automated Heuristic Design (AHD) using Large Language Models (LLMs) has achieved notable success in recent years. Despite the effectiveness of existing approaches, they only design a single heuristic to serve all problem instances, often inducing poor generalization across different distributions or settings. To address this issue, we propose Automated Heuristic Set Design (AHSD), a new formulation for LLM-driven AHD. The aim of AHSD is to automatically generate a small-sized complementary heuristic set to serve diverse problem instances, such that each problem instance could be optimized by at least one heuristic in this set. We show that the objective function of AHSD is monotone and supermodular. Then, we propose Evolution of Heuristic Set (EoH-S) to apply the AHSD formulation for LLM-driven AHD. With two novel mechanisms of complementary population management and complementary-aware memetic search, EoH-S could effectively generate a set of high-quality and complementary heuristics. Comprehensive experimental results on three AHD tasks with diverse instances spanning various sizes and distributions demonstrate that EoH-S consistently outperforms existing state-of-the-art AHD methods and achieves up to 60\% performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03080v1">ContractEval: Benchmarking LLMs for Clause-Level Legal Risk Identification in Commercial Contracts</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      The potential of large language models (LLMs) in specialized domains such as legal risk analysis remains underexplored. In response to growing interest in locally deploying open-source LLMs for legal tasks while preserving data confidentiality, this paper introduces ContractEval, the first benchmark to thoroughly evaluate whether open-source LLMs could match proprietary LLMs in identifying clause-level legal risks in commercial contracts. Using the Contract Understanding Atticus Dataset (CUAD), we assess 4 proprietary and 15 open-source LLMs. Our results highlight five key findings: (1) Proprietary models outperform open-source models in both correctness and output effectiveness, though some open-source models are competitive in certain specific dimensions. (2) Larger open-source models generally perform better, though the improvement slows down as models get bigger. (3) Reasoning ("thinking") mode improves output effectiveness but reduces correctness, likely due to over-complicating simpler tasks. (4) Open-source models generate "no related clause" responses more frequently even when relevant clauses are present. This suggests "laziness" in thinking or low confidence in extracting relevant content. (5) Model quantization speeds up inference but at the cost of performance drop, showing the tradeoff between efficiency and accuracy. These findings suggest that while most LLMs perform at a level comparable to junior legal assistants, open-source models require targeted fine-tuning to ensure correctness and effectiveness in high-stakes legal settings. ContractEval offers a solid benchmark to guide future development of legal-domain LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14220v2">Enhancing Spectral Graph Neural Networks with LLM-Predicted Homophily</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Spectral Graph Neural Networks (SGNNs) have achieved remarkable performance in tasks such as node classification due to their ability to learn flexible filters. Typically, these filters are learned under the supervision of downstream tasks, enabling SGNNs to adapt to diverse structural patterns. However, in scenarios with limited labeled data, SGNNs often struggle to capture the optimal filter shapes, resulting in degraded performance, especially on graphs with heterophily. Meanwhile, the rapid progress of Large Language Models (LLMs) has opened new possibilities for enhancing graph learning without modifying graph structure or requiring task-specific training. In this work, we propose a novel framework that leverages LLMs to estimate the homophily level of a graph and uses this global structural prior to guide the construction of spectral filters. Specifically, we design a lightweight and plug-and-play pipeline where a small set of labeled node pairs is formatted as natural language prompts for the LLM, which then predicts the graph's homophily ratio. This estimated value informs the spectral filter basis, enabling SGNNs to adapt more effectively to both homophilic and heterophilic structures. Extensive experiments on multiple benchmark datasets demonstrate that our LLM-assisted spectral framework consistently improves performance over strong SGNN baselines. Importantly, this enhancement incurs negligible computational and monetary cost, making it a practical solution for real-world graph applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09037v3">A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 72 pages, 6 figures. Accepted to TMLR, with Survey Certification award
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02999v1">AGENTiGraph: A Multi-Agent Knowledge Graph Framework for Interactive, Domain-Specific LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 CIKM 2025, Demo Track
    </div>
    <details class="paper-abstract">
      AGENTiGraph is a user-friendly, agent-driven system that enables intuitive interaction and management of domain-specific data through the manipulation of knowledge graphs in natural language. It gives non-technical users a complete, visual solution to incrementally build and refine their knowledge bases, allowing multi-round dialogues and dynamic updates without specialized query languages. The flexible design of AGENTiGraph, including intent classification, task planning, and automatic knowledge integration, ensures seamless reasoning between diverse tasks. Evaluated on a 3,500-query benchmark within an educational scenario, the system outperforms strong zero-shot baselines (achieving 95.12% classification accuracy, 90.45% execution success), indicating potential scalability to compliance-critical or multi-step queries in legal and medical domains, e.g., incorporating new statutes or research on the fly. Our open-source demo offers a powerful new paradigm for multi-turn enterprise knowledge management that bridges LLMs and structured graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02994v1">When AIs Judge AIs: The Rise of Agent-as-a-Judge Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in capability and autonomy, evaluating their outputs-especially in open-ended and complex tasks-has become a critical bottleneck. A new paradigm is emerging: using AI agents as the evaluators themselves. This "agent-as-a-judge" approach leverages the reasoning and perspective-taking abilities of LLMs to assess the quality and safety of other models, promising calable and nuanced alternatives to human evaluation. In this review, we define the agent-as-a-judge concept, trace its evolution from single-model judges to dynamic multi-agent debate frameworks, and critically examine their strengths and shortcomings. We compare these approaches across reliability, cost, and human alignment, and survey real-world deployments in domains such as medicine, law, finance, and education. Finally, we highlight pressing challenges-including bias, robustness, and meta evaluation-and outline future research directions. By bringing together these strands, our review demonstrates how agent-based judging can complement (but not replace) human oversight, marking a step toward trustworthy, scalable evaluation for next-generation LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02979v1">Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2507.10593
    </div>
    <details class="paper-abstract">
      The proliferation of tool-augmented Large Language Models (LLMs) has created a fragmented ecosystem where developers must navigate multiple protocols, manual schema definitions, and complex execution workflows. We address this challenge by proposing a unified approach to tool integration that abstracts protocol differences while optimizing execution performance. Our solution demonstrates how protocol-agnostic design principles can significantly reduce development overhead through automated schema generation, dual-mode concurrent execution, and seamless multi-source tool management. Experimental results show 60-80% code reduction across integration scenarios, performance improvements up to 3.1x through optimized concurrency, and full compatibility with existing function calling standards. This work contributes both theoretical insights into tool integration architecture and practical solutions for real-world LLM application development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03966v1">GP and LLMs for Program Synthesis: No Clear Winners</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Genetic programming (GP) and large language models (LLMs) differ in how program specifications are provided: GP uses input-output examples, and LLMs use text descriptions. In this work, we compared the ability of PushGP and GPT-4o to synthesize computer programs for tasks from the PSB2 benchmark suite. We used three prompt variants with GPT-4o: input-output examples (data-only), textual description of the task (text-only), and a combination of both textual descriptions and input-output examples (data-text). Additionally, we varied the number of input-output examples available for building programs. For each synthesizer and task combination, we compared success rates across all program synthesizers, as well as the similarity between successful GPT-4o synthesized programs. We found that the combination of PushGP and GPT-4o with data-text prompting led to the greatest number of tasks solved (23 of the 25 tasks), even though several tasks were solved exclusively by only one of the two synthesizers. We also observed that PushGP and GPT-4o with data-only prompting solved fewer tasks with the decrease in the training set size, while the remaining synthesizers saw no decrease. We also detected significant differences in similarity between the successful programs synthesized for GPT-4o with text-only and data-only prompting. With there being no dominant program synthesizer, this work highlights the importance of different optimization techniques used by PushGP and LLMs to synthesize programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03931v1">Analyzing Prominent LLMs: An Empirical Study of Performance and Complexity in Solving LeetCode Problems</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 11 pages, 13 figures, 29th International Conference on Evaluation and Assessment in Software Engineering (EASE)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like ChatGPT, Copilot, Gemini, and DeepSeek are transforming software engineering by automating key tasks, including code generation, testing, and debugging. As these models become integral to development workflows, a systematic comparison of their performance is essential for optimizing their use in real world applications. This study benchmarks these four prominent LLMs on one hundred and fifty LeetCode problems across easy, medium, and hard difficulties, generating solutions in Java and Python. We evaluate each model based on execution time, memory usage, and algorithmic complexity, revealing significant performance differences. ChatGPT demonstrates consistent efficiency in execution time and memory usage, while Copilot and DeepSeek show variability as task complexity increases. Gemini, although effective on simpler tasks, requires more attempts as problem difficulty rises. Our findings provide actionable insights into each model's strengths and limitations, offering guidance for developers selecting LLMs for specific coding tasks and providing insights on the performance and complexity of GPT-like generated solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15715v2">From Queries to Criteria: Understanding How Astronomers Evaluate LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09516v5">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12422v2">FactEHR: A Dataset for Evaluating Factuality in Clinical Notes Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 To appear at MLHC 2025
    </div>
    <details class="paper-abstract">
      Verifying and attributing factual claims is essential for the safe and effective use of large language models (LLMs) in healthcare. A core component of factuality evaluation is fact decomposition, the process of breaking down complex clinical statements into fine-grained atomic facts for verification. Recent work has proposed fact decomposition, which uses LLMs to rewrite source text into concise sentences conveying a single piece of information, to facilitate fine-grained fact verification. However, clinical documentation poses unique challenges for fact decomposition due to dense terminology and diverse note types and remains understudied. To address this gap and explore these challenges, we present FactEHR, an NLI dataset consisting of document fact decompositions for 2,168 clinical notes spanning four types from three hospital systems, resulting in 987,266 entailment pairs. We assess the generated facts on different axes, from entailment evaluation of LLMs to a qualitative analysis. Our evaluation, including review by the clinicians, reveals substantial variability in LLM performance for fact decomposition. For example, Gemini-1.5-Flash consistently generates relevant and accurate facts, while Llama-3 8B produces fewer and less consistent outputs. The results underscore the need for better LLM capabilities to support factual verification in clinical text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14805v2">How Well Do LLMs Represent Values Across Cultures? Empirical Analysis of LLM Responses Based on Hofstede Cultural Dimensions</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 KDD 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) attempt to imitate human behavior by responding to humans in a way that pleases them, including by adhering to their values. However, humans come from diverse cultures with different values. It is critical to understand whether LLMs showcase different values to the user based on the stereotypical values of a user's known country. We prompt different LLMs with a series of advice requests based on 5 Hofstede Cultural Dimensions -- a quantifiable way of representing the values of a country. Throughout each prompt, we incorporate personas representing 36 different countries and, separately, languages predominantly tied to each country to analyze the consistency in the LLMs' cultural understanding. Through our analysis of the responses, we found that LLMs can differentiate between one side of a value and another, as well as understand that countries have differing values, but will not always uphold the values when giving advice, and fail to understand the need to answer differently based on different cultural values. Rooted in these findings, we present recommendations for training value-aligned and culturally sensitive LLMs. More importantly, the methodology and the framework developed here can help further understand and mitigate culture and language alignment issues with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03793v1">AttnTrace: Attention-based Context Traceback for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 The code is available at https://github.com/Wang-Yanting/AttnTrace. The demo is available at https://huggingface.co/spaces/SecureLLMSys/AttnTrace
    </div>
    <details class="paper-abstract">
      Long-context large language models (LLMs), such as Gemini-2.5-Pro and Claude-Sonnet-4, are increasingly used to empower advanced AI systems, including retrieval-augmented generation (RAG) pipelines and autonomous agents. In these systems, an LLM receives an instruction along with a context--often consisting of texts retrieved from a knowledge database or memory--and generates a response that is contextually grounded by following the instruction. Recent studies have designed solutions to trace back to a subset of texts in the context that contributes most to the response generated by the LLM. These solutions have numerous real-world applications, including performing post-attack forensic analysis and improving the interpretability and trustworthiness of LLM outputs. While significant efforts have been made, state-of-the-art solutions such as TracLLM often lead to a high computation cost, e.g., it takes TracLLM hundreds of seconds to perform traceback for a single response-context pair. In this work, we propose AttnTrace, a new context traceback method based on the attention weights produced by an LLM for a prompt. To effectively utilize attention weights, we introduce two techniques designed to enhance the effectiveness of AttnTrace, and we provide theoretical insights for our design choice. We also perform a systematic evaluation for AttnTrace. The results demonstrate that AttnTrace is more accurate and efficient than existing state-of-the-art context traceback methods. We also show that AttnTrace can improve state-of-the-art methods in detecting prompt injection under long contexts through the attribution-before-detection paradigm. As a real-world application, we demonstrate that AttnTrace can effectively pinpoint injected instructions in a paper designed to manipulate LLM-generated reviews. The code is at https://github.com/Wang-Yanting/AttnTrace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03771v1">Trustworthiness of Legal Considerations for the Use of LLMs in Education</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 11 pages, 3 figures, 6 tables
    </div>
    <details class="paper-abstract">
      As Artificial Intelligence (AI), particularly Large Language Models (LLMs), becomes increasingly embedded in education systems worldwide, ensuring their ethical, legal, and contextually appropriate deployment has become a critical policy concern. This paper offers a comparative analysis of AI-related regulatory and ethical frameworks across key global regions, including the European Union, United Kingdom, United States, China, and Gulf Cooperation Council (GCC) countries. It maps how core trustworthiness principles, such as transparency, fairness, accountability, data privacy, and human oversight are embedded in regional legislation and AI governance structures. Special emphasis is placed on the evolving landscape in the GCC, where countries are rapidly advancing national AI strategies and education-sector innovation. To support this development, the paper introduces a Compliance-Centered AI Governance Framework tailored to the GCC context. This includes a tiered typology and institutional checklist designed to help regulators, educators, and developers align AI adoption with both international norms and local values. By synthesizing global best practices with region-specific challenges, the paper contributes practical guidance for building legally sound, ethically grounded, and culturally sensitive AI systems in education. These insights are intended to inform future regulatory harmonization and promote responsible AI integration across diverse educational environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04721v1">Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      We introduce a low-latency telecom AI voice agent pipeline for real-time, interactive telecommunications use, enabling advanced voice AI for call center automation, intelligent IVR (Interactive Voice Response), and AI-driven customer support. The solution is built for telecom, combining four specialized models by NetoAI: TSLAM, a 4-bit quantized Telecom-Specific Large Language Model (LLM); T-VEC, a Telecom-Specific Embedding Model; TTE, a Telecom-Specific Automatic Speech Recognition (ASR) model; and T-Synth, a Telecom-Specific Text-to-Speech (TTS) model. These models enable highly responsive, domain-adapted voice AI agents supporting knowledge-grounded spoken interactions with low latency. The pipeline integrates streaming ASR (TTE), conversational intelligence (TSLAM), retrieval augmented generation (RAG) over telecom documents, and real-time TTS (T-Synth), setting a new benchmark for telecom voice assistants. To evaluate the system, we built a dataset of 500 human-recorded telecom questions from RFCs, simulating real telecom agent queries. This framework allows analysis of latency, domain relevance, and real-time performance across the stack. Results show that TSLAM, TTE, and T-Synth deliver real-time factors (RTF) below 1.0, supporting enterprise, low-latency telecom deployments. These AI agents -- powered by TSLAM, TTE, and T-Synth -- provide a foundation for next-generation telecom AI, enabling automated customer support, diagnostics, and more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04720v1">Who is a Better Player: LLM against LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Adversarial board games, as a paradigmatic domain of strategic reasoning and intelligence, have long served as both a popular competitive activity and a benchmark for evaluating artificial intelligence (AI) systems. Building on this foundation, we propose an adversarial benchmarking framework to assess the comprehensive performance of Large Language Models (LLMs) through board games competition, compensating the limitation of data dependency of the mainstream Question-and-Answer (Q&A) based benchmark method. We introduce Qi Town, a specialized evaluation platform that supports 5 widely played games and involves 20 LLM-driven players. The platform employs both the Elo rating system and a novel Performance Loop Graph (PLG) to quantitatively evaluate the technical capabilities of LLMs, while also capturing Positive Sentiment Score (PSS) throughout gameplay to assess mental fitness. The evaluation is structured as a round-robin tournament, enabling systematic comparison across players. Experimental results indicate that, despite technical differences, most LLMs remain optimistic about winning and losing, demonstrating greater adaptability to high-stress adversarial environments than humans. On the other hand, the complex relationship between cyclic wins and losses in PLGs exposes the instability of LLMs' skill play during games, warranting further explanation and exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03686v1">CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Technical Report; 31 Pages
    </div>
    <details class="paper-abstract">
      Answer verification is crucial not only for evaluating large language models (LLMs) by matching their unstructured outputs against standard answers, but also serves as the reward model to guide LLM optimization. Most evaluation frameworks rely on regularized matching or employ general LLMs for answer verification, which demands extensive, repetitive customization for regex rules or evaluation prompts. Two fundamental limitations persist in current methodologies: 1) the absence of comprehensive benchmarks that systematically evaluate verification capabilities across different LLMs; and 2) the nascent stage of verifier development, where existing approaches lack both the robustness to handle complex edge cases and the generalizability across different domains. In this work, we develop CompassVerifier, an accurate and robust lightweight verifier model for evaluation and outcome reward. It demonstrates multi-domain competency spanning math, knowledge, and diverse reasoning tasks, with the capability to process various answer types, including multi-subproblems, formulas, and sequence answers, while effectively identifying abnormal/invalid responses. We introduce VerifierBench benchmark comprising model outputs collected from multiple data sources, augmented through manual analysis of metaerror patterns to enhance CompassVerifier. We anticipate that CompassVerifier and VerifierBench will facilitate answer verification, evaluation protocols, and reinforcement learning research. Code and dataset are available at https://github.com/open-compass/CompassVerifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03685v1">No LLM Solved Yu Tsumura's 554th Problem</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 67 pages
    </div>
    <details class="paper-abstract">
      We show, contrary to the optimism about LLM's problem-solving abilities, fueled by the recent gold medals that were attained, that a problem exists -- Yu Tsumura's 554th problem -- that a) is within the scope of an IMO problem in terms of proof sophistication, b) is not a combinatorics problem which has caused issues for LLMs, c) requires fewer proof techniques than typical hard IMO problems, d) has a publicly available solution (likely in the training data of LLMs), and e) that cannot be readily solved by any existing off-the-shelf LLM (commercial or open-source).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03678v1">More Than a Score: Probing the Impact of Prompt Specificity on LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      State-of-the-art Large Language Models (LLMs) achieve high pass@1 on general benchmarks like HumanEval but underperform on specialized suites such as ParEval. Is this due to LLMs missing domain knowledge or insufficient prompt detail is given? To answer this, we introduce PartialOrderEval, which augments any code generation benchmark with a partial order of prompts from minimal to maximally detailed. Applying it to HumanEval and both serial and OpenMP subsets of ParEval, we measure how pass@1 scales with prompt specificity. Our experiments with Llama-3.x and Qwen2.5-Coder demonstrate varying degrees of prompt sensitivity across different tasks, and a qualitative analysis highlights explicit I/O specifications, edge-case handling, and stepwise breakdowns as the key drivers of prompt detail improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v2">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address for distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03628v1">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations at eBay</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Sellers at eBay are recommended keyphrases to bid on to enhance the performance of their advertising campaigns. The relevance of these keyphrases is crucial in avoiding the overcrowding of search systems with irrelevant items and maintaining a positive seller perception. It is essential that keyphrase recommendations align with both seller and Search judgments regarding auctions. Due to the difficulty in procuring negative human judgment at scale, employing LLM-as-a-judge to mimic seller judgment has been established as the norm in several studies. This study introduces a novel two-step LLM distillation process from a LLM-judge used to debias our Embedding Based Retrieval (EBR) model from the various biases that exist in click-data. We distill from an LLM teacher via a cross-encoder assistant into a bi-encoder student using a multi-task training approach, ultimately employing the student bi-encoder to retrieve relevant advertiser keyphrases. We show that integrating a knowledge distillation process from LLMs in a multi-task training setup enhances bi-encoder performance in retrieving relevant advertiser keyphrases at eBay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v4">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03622v1">Refining Critical Thinking in LLM Code Generation: A Faulty Premise-based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      With the advancement of code generation capabilities in large language models (LLMs), their reliance on input premises has intensified. When users provide inputs containing faulty premises, the probability of code generation hallucinations rises significantly, exposing deficiencies in their self-scrutiny capabilities. This paper proposes Faulty Premises Bench (FPBench), the first code generation evaluation framework targeting faulty premises. By systematically constructing three categories of faulty premises and integrating multi-dimensional evaluation metrics, it conducts in-depth assessments of 15 representative LLMs. The key findings are as follows: (1) Most models exhibit poor reasoning abilities and suboptimal code generation performance under faulty premises, heavily relying on explicit prompts for error detection, with limited self-scrutiny capabilities; (2) Faulty premises trigger a point of diminishing returns in resource investment, leading to blindly increasing length fails to enhance quality; (3) The three types of faulty premises respectively activate distinct defect patterns in models, revealing a triple dissociation in the cognitive mechanisms of code generation models. This study not only highlights the urgent need for LLMs to proactively verify premises in code generation but also, through the proposed FPBench framework and multi-dimensional evaluation system, provides a theoretical foundation and practical pathway for developing reliable, human-centric code generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03611v1">Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 12 pages, 8 figures excluding appendix
    </div>
    <details class="paper-abstract">
      This paper presents Block, a distributed scheduling framework designed to optimize load balancing and auto-provisioning across instances in large language model serving frameworks by leveraging contextual information from incoming requests. Unlike popular model serving systems that rely on monolithic and heuristic task schedulers, Block operates as a fully distributed, stateless, and predictive scheduling system to achieve low overhead, reliability, and scalability. It leverages the deterministic and predictable characteristics of LLM inferences, such as host configurations, response lengths, and hardware performance, to make scheduling decisions based on accurately predicted metrics. Evaluation on a 12 GPUs cluster shows that Block significantly outperforms heuristic schedulers, boosting serving capacity by up to 16.7\% and reducing P99 tail latency by up to 49.5\%. These performance gains remain consistent across diverse models, workloads and configurations. Code and data are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v4">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03603v1">ReFuzzer: Feedback-Driven Approach to Enhance Validity of LLM-Generated Test Programs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Existing LLM-based compiler fuzzers often produce syntactically or semantically invalid test programs, limiting their effectiveness in exercising compiler optimizations and backend components. We introduce ReFuzzer, a framework for refining LLM-generated test programs by systematically detecting and correcting compilation and runtime violations (e.g. division by zero or array out-of-bounds accesses). ReFuzzer employs a feedback loop with a local LLM to validate and filter erroneous programs before execution, improving fuzzing effectiveness beyond crash detection and enabling the generation of diverse yet valid test programs. We evaluated ReFuzzer's effectiveness across black-, grey- and white-box fuzzing approaches targeting LLVM/Clang. ReFuzzer improved test programs' validity from 47.0-49.4% to 96.6-97.3%, with an average processing time of 2.9-3.5 s per test program on a dual-GPU machine. Further, refuzzing significantly increased code coverage in critical optimization and IR generation components. For example, vectorization coverage had an absolute improvement of 9.2%, 2.3%, and 7.1% in black-, grey-, and white-box fuzzing, enhancing testing effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06219v2">Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 COLM 2025 Camera Ready version
    </div>
    <details class="paper-abstract">
      The increasing adoption of web crawling opt-outs by copyright holders of online content raises critical questions about the impact of data compliance on large language model (LLM) performance. However, little is known about how these restrictions (and the resultant filtering of pretraining datasets) affect the capabilities of models trained using these corpora. In this work, we conceptualize this effect as the $\textit{data compliance gap}$ (DCG), which quantifies the performance difference between models trained on datasets that comply with web crawling opt-outs, and those that do not. We measure the data compliance gap in two settings: pretraining models from scratch and continual pretraining from existing compliant models (simulating a setting where copyrighted data could be integrated later in pretraining). Our experiments with 1.5B models show that, as of January 2025, compliance with web data opt-outs does not degrade general knowledge acquisition (close to 0\% DCG). However, in specialized domains such as biomedical research, excluding major publishers leads to performance declines. These findings suggest that while general-purpose LLMs can be trained to perform equally well using fully open data, performance in specialized domains may benefit from access to high-quality copyrighted sources later in training. Our study provides empirical insights into the long-debated trade-off between data compliance and downstream model performance, informing future discussions on AI training practices and policy decisions. Our website is available at https://data-compliance.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03571v1">Tackling Distribution Shift in LLM via KILO: Knowledge-Instructed Learning for Continual Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from performance degradation when faced with domain shifts, primarily due to catastrophic forgetting. In this work, we propose KILO (Knowledge-Instructed Learning for Continual Adaptation), a novel continual learning framework that integrates dynamic knowledge graphs with instruction tuning. By leveraging retrieved domain-specific knowledge as guidance during training, KILO enhances both adaptability to new domains and retention of previously acquired knowledge. We pretrain our model on WikiText-103 and evaluate sequential adaptation across four diverse target domains: BioASQ, SciQ, TweetEval, and MIND. Our experiments demonstrate that KILO consistently outperforms strong baselines, including continual fine-tuning, ERNIE 2.0, and CPT, in terms of backward transfer, forward transfer, F1 score, retention rate, and training efficiency. These results highlight the effectiveness of combining structured knowledge retrieval and instruction prompting to overcome domain shift challenges in continual learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03558v1">SAGE-HLS: Syntax-Aware AST-Guided LLM for High-Level Synthesis Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted to the IEEE International Conference on Computer Design (ICCD 2025)
    </div>
    <details class="paper-abstract">
      In today's rapidly evolving field of electronic design automation (EDA), the complexity of hardware designs is increasing, necessitating more sophisticated automation solutions. High-level synthesis (HLS), as a pivotal solution, automates hardware designs from high-level abstractions (e.g., C/C++). However, it faces significant challenges, particularly in design space exploration and optimization. While large language models (LLMs) have shown notable capabilities in code generation, their application to HLS has been limited due to the scarcity of (publicly) available HLS code datasets. Hence, research in this domain has primarily focused on techniques such as prompt engineering and retrieval-augmented generation (RAG). To overcome this limitation, this paper introduces SAGE-HLS, the first-of-its-kind fine-tuned LLM specifically for HLS code generation. Our method includes three key advancements: (i) We implement Verilog-to-C/C++ porting, converting verified and synthesizable Verilog codes into corresponding C, creating a dataset of 16.7K HLS codes; (ii) We implement a fine-tuning strategy, which is based on instruction prompting to code generation guided by abstract syntax tree (AST); (iii) We develop a semi-automated evaluation framework using VerilogEval to assess the functionality of the generated HLS code. Our experiments show that SAGE-HLS, fined-tuned on the QwenCoder (2.5) 7B model, achieves a near 100% success rate in code synthesizability and a 75% success rate in functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v1">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using large language models, a paradigm known as "LLMas-a-judge." However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and taskrelevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a lightweight and efficient framework for enhancing LLM-as-a-Judge alignment with human scoring, via internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer scoretoken logits and computing the expected score from a softmax-based distribution, with the LLM backbone kept frozen. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03547v1">Guided Reality: Generating Visually-Enriched AR Task Guidance with LLMs and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 To appear at UIST 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the automatic generation of step-by-step augmented reality (AR) instructions for a wide range of physical tasks. However, existing LLM-based AR guidance often lacks rich visual augmentations to effectively embed instructions into spatial context for a better user understanding. We present Guided Reality, a fully automated AR system that generates embedded and dynamic visual guidance based on step-by-step instructions. Our system integrates LLMs and vision models to: 1) generate multi-step instructions from user queries, 2) identify appropriate types of visual guidance, 3) extract spatial information about key interaction points in the real world, and 4) embed visual guidance in physical space to support task execution. Drawing from a corpus of user manuals, we define five categories of visual guidance and propose an identification strategy based on the current step. We evaluate the system through a user study (N=16), completing real-world tasks and exploring the system in the wild. Additionally, four instructors shared insights on how Guided Reality could be integrated into their training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03523v1">FilBench: Can LLMs Understand and Generate Filipino?</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of LLMs on English-based tasks, little is known about their capabilities in specific languages such as Filipino. In this work, we address this gap by introducing FilBench, a Filipino-centric benchmark designed to evaluate LLMs across a diverse set of tasks and capabilities in Filipino, Tagalog, and Cebuano. We carefully curate the tasks in FilBench to reflect the priorities and trends of NLP research in the Philippines such as Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation. By evaluating 27 state-of-the-art LLMs on FilBench, we find that several LLMs suffer from reading comprehension and translation capabilities. Our results indicate that FilBench is challenging, with the best model, GPT-4o, achieving only a score of 72.23%. Moreover, we also find that models trained specifically for Southeast Asian languages tend to underperform on FilBench, with the highest-performing model, SEA-LION v3 70B, achieving only a score of 61.07%. Our work demonstrates the value of curating language-specific LLM benchmarks to aid in driving progress on Filipino NLP and increasing the inclusion of Philippine languages in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17432v2">Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 13 pages, 8 figures, Accepted by ACM MM2025, Project page: https://garygutc.github.io/UniME
    </div>
    <details class="paper-abstract">
      The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored.In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLM\'s language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short and long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03487v1">BitsAI-Fix: LLM-Driven Approach for Automated Lint Error Resolution in Practice</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      As enterprise codebases continue to grow in scale and complexity, the volume of lint errors far exceeds engineers' manual remediation capacity, leading to continuous accumulation of technical debt and hindered development efficiency. This paper presents BitsAI-Fix, an automated lint error remediation workflow based on Large Language Models (LLMs), designed to address this critical challenge in industrial-scale environments. BitsAI-Fix employs tree-sitter for context expansion and generates search-and-replace format patches through specially trained LLMs, followed by lint scan re-verification to output final remediation results. Additionally, our approach introduces an innovative progressive reinforcement learning (RL) training strategy that can automatically acquire verifiable training data during the project cold-start phase and continuously iterate the model by collecting online samples through feedback after system deployment. Furthermore, we designed a targeted rule-based reward mechanism that combines format rewards and correctness rewards while penalizing redundant modifications. We also propose a "code diff matching" methodology to continuously track online effectiveness. In production deployment at ByteDance, our solution has supported over 5,000 engineers, resolved more than 12,000 static analysis issues, achieved approximately 85% remediation accuracy, with around 1,000 weekly active adopters. This work demonstrates the practical feasibility of LLM-based code remediation solutions in enterprise environments and serves as a reference for automated code fix in large-scale industrial scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03464v1">Learning to Incentivize: LLM-Empowered Contract for AIGC Offloading in Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      With the rapid growth in demand for AI-generated content (AIGC), edge AIGC service providers (ASPs) have become indispensable. However, designing incentive mechanisms that motivate ASPs to deliver high-quality AIGC services remains a challenge, especially in the presence of information asymmetry. In this paper, we address bonus design between a teleoperator and an edge ASP when the teleoperator cannot observe the ASP's private settings and chosen actions (diffusion steps). We formulate this as an online learning contract design problem and decompose it into two subproblems: ASP's settings inference and contract derivation. To tackle the NP-hard setting-inference subproblem with unknown variable sizes, we introduce a large language model (LLM)-empowered framework that iteratively refines a naive seed solver using the LLM's domain expertise. Upon obtaining the solution from the LLM-evolved solver, we directly address the contract derivation problem using convex optimization techniques and obtain a near-optimal contract. Simulation results on our Unity-based teleoperation platform show that our method boosts the teleoperator's utility by $5 \sim 40\%$ compared to benchmarks, while preserving positive incentives for the ASP. The code is available at https://github.com/Zijun0819/llm4contract.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06787v4">Bridging LLMs and KGs without Fine-Tuning: Intermediate Probing Meets Subgraph-Aware Entity Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Traditional knowledge graph completion (KGC) methods rely solely on structural information, struggling with the inherent sparsity of knowledge graphs (KGs). By contrast, Large Language Models (LLMs) encapsulate extensive world knowledge and exhibit powerful context modeling capabilities, making them promising for mitigating the limitations of traditional methods. However, direct fine-tuning of LLMs for KGC, though effective, imposes substantial computational and memory overheads, while utilizing non-fine-tuned LLMs is efficient but yields suboptimal performance. In this work, we propose a novel framework that synergizes the strengths of LLMs with robust knowledge representation to enable effective and efficient KGC. We extract the context-aware hidden states of knowledge triples from the intermediate layers of LLMs, thereby capturing rich semantic and relational nuances. These representations are then utilized to train a data-efficient classifier tailored specifically for KGC tasks. To bridge the semantic gaps between LLMs and KGs, we employ subgraph sampling on KGs to generate model-friendly entity descriptions. We further adopt sliced mutual information (SMI) as a principled metric to quantify the task-specific information encoded in these representations. Extensive experiments on standard benchmarks validate the efficiency and effectiveness of our approach. We achieve a 47\% relative improvement over previous methods based on non-fine-tuned LLMs and, to our knowledge, are the first to achieve classification performance comparable to fine-tuned LLMs while enhancing GPU memory efficiency by $188\times$ and accelerating training and inference by $26.11\times$.
    </details>
</div>
