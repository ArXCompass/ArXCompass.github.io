# llm - 2025_10

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11696v1">QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Code is available at https://github.com/NVlabs/QeRL
    </div>
    <details class="paper-abstract">
      We propose QeRL, a Quantization-enhanced Reinforcement Learning framework for large language models (LLMs). While RL is essential for LLMs' reasoning capabilities, it is resource-intensive, requiring substantial GPU memory and long rollout durations. QeRL addresses these issues by combining NVFP4 quantization with Low-Rank Adaptation (LoRA), accelerating rollout phase of RL while reducing memory overhead. Beyond efficiency, our findings show that quantization noise increases policy entropy, enhancing exploration, and enabling the discovery of better strategies during RL. To further optimize exploration, QeRL introduces an Adaptive Quantization Noise (AQN) mechanism, which dynamically adjusts noise during training. Experiments demonstrate that QeRL delivers over 1.5 times speedup in the rollout phase. Moreover, this is the first framework to enable RL training of a 32B LLM on a single H100 80GB GPU, while delivering overall speedups for RL training. It also achieves faster reward growth and higher final accuracy than 16-bit LoRA and QLoRA, while matching the performance of full-parameter fine-tuning on mathematical benchmarks such as GSM8K (90.8%) and MATH 500 (77.4%) in the 7B model. These results establish QeRL as an efficient and effective framework for RL training in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11695v1">When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v4">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning (NFL-HR)**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the NL-FL Problem Alignment method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the Mixed Problem Input technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based Answer Extraction mechanism. Comprehensive experiments demonstrate that the NFL-HR framework achieves **89.80**% and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by **4.60%** and **4.82%**, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11615v1">LLM-Oriented Token-Adaptive Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Knowledge distillation (KD) is a key technique for compressing large-scale language models (LLMs), yet prevailing logit-based methods typically employ static strategies that are misaligned with the dynamic learning process of student models. These methods typically treat all tokens indiscriminately and apply a single, fixed temperature, resulting in suboptimal knowledge transfer. To address these limitations, we propose LLM-Oriented Token-Adaptive Knowledge Distillation (AdaKD), a novel framework that adapts the distillation process to the real-time learning state of each token. AdaKD consists of two synergistic modules driven by a unified token difficulty metric. First, our Loss-Driven Adaptive Token Focusing (LATF) module dynamically adjusts the distillation focus by monitoring the student's learning stability, concentrating computational resources on the most valuable tokens at each training phase. Second, we introduce Inverse Difficulty Temperature Scaling (IDTS), a counterintuitive yet effective token-level temperature strategy. It employs low temperatures for difficult tokens for targeted error correction, and high temperatures for easy tokens to encourage students to learn from the teacher's complete and smooth output distribution, thereby enhancing generalization. As a plug-and-play framework, AdaKD can consistently improve the performance of various distillation methods on multiple model architectures and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11588v1">Analyzing and Internalizing Complex Policy Documents for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems rely on in-context policy documents encoding diverse business rules. As requirements grow, these documents expand rapidly, causing high computational overhead. This motivates developing internalization methods that embed policy documents into model priors while preserving performance. Prior prompt compression work targets generic prompts, but agentic policy documents span multiple complexity levels and require deeper reasoning, making internalization harder. We introduce CC-Gen, an agentic benchmark generator with Controllable Complexity across four levels, enabling systematic evaluation of agents' ability to handle complexity and offering a unified framework for assessing policy internalization. Our analysis shows that complex policy specifications governing workflows pose major reasoning challenges. Supporting internalization with gold user agent interaction trajectories containing chain-of-thought (CoT) annotations via supervised fine-tuning (SFT) is data-intensive and degrades sharply as policy complexity increases. To mitigate data and reasoning burdens, we propose Category-Aware Policy Continued Pretraining (CAP-CPT). Our automated pipeline parses policy documents to extract key specifications, grouping them into factual, behavioral, and conditional categories, and isolating complex conditions that drive workflow complexity. This guides targeted data synthesis and enables agents to internalize policy information through an autoregressive pretraining loss. Experiments show CAP-CPT improves SFT baselines in all settings, with up to 41% and 22% gains on Qwen-3-32B, achieving 97.3% prompt length reduction on CC-Gen and further enhancing tau-Bench with minimal SFT data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11563v1">Culturally-Aware Conversations: A Framework & Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 To appear at the 4th HCI + NLP Workshop @ EMNLP
    </div>
    <details class="paper-abstract">
      Existing benchmarks that measure cultural adaptation in LLMs are misaligned with the actual challenges these models face when interacting with users from diverse cultural backgrounds. In this work, we introduce the first framework and benchmark designed to evaluate LLMs in realistic, multicultural conversational settings. Grounded in sociocultural theory, our framework formalizes how linguistic style - a key element of cultural communication - is shaped by situational, relational, and cultural context. We construct a benchmark dataset based on this framework, annotated by culturally diverse raters, and propose a new set of desiderata for cross-cultural evaluation in NLP: conversational framing, stylistic sensitivity, and subjective correctness. We evaluate today's top LLMs on our benchmark and show that these models struggle with cultural adaptation in a conversational setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04310v3">EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07772v2">An approach for systematic decomposition of complex llm tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from reliability issues on complex tasks, as existing decomposition methods are heuristic and rely on agent or manual decomposition. This work introduces a novel, systematic decomposition framework that we call Analysis of CONstraint-Induced Complexity (ACONIC), which models the task as a constraint problem and leveraging formal complexity measures to guide decomposition. On combinatorial (SATBench) and LLM database querying tasks (Spider), we find that by decomposing the tasks following the measure of complexity, agent can perform considerably better (10-40 percentage point).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11558v1">Zero Data Retention in LLM-based Enterprise AI Assistants: A Comparative Study of Market Leading Agentic AI Products</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Governance of data, compliance, and business privacy matters, particularly for healthcare and finance businesses. Since the recent emergence of AI enterprise AI assistants enhancing business productivity, safeguarding private data and compliance is now a priority. With the implementation of AI assistants across the enterprise, the zero data retention can be achieved by implementing zero data retention policies by Large Language Model businesses like Open AI and Anthropic and Meta. In this work, we explore zero data retention policies for the Enterprise apps of large language models (LLMs). Our key contribution is defining the architectural, compliance, and usability trade-offs of such systems in parallel. In this research work, we examine the development of commercial AI assistants with two industry leaders and market titans in this arena - Salesforce and Microsoft. Both of these companies used distinct technical architecture to support zero data retention policies. Salesforce AgentForce and Microsoft Copilot are among the leading AI assistants providing much-needed push to business productivity in customer care. The purpose of this paper is to analyze the technical architecture and deployment of zero data retention policy by consuming applications as well as big language models service providers like Open Ai, Anthropic, and Meta.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11557v1">Invisible Languages of the LLM Universe</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models are trained on massive multilingual corpora, yet this abundance masks a profound crisis: of the world's 7,613 living languages, approximately 2,000 languages with millions of speakers remain effectively invisible in digital ecosystems. We propose a critical framework connecting empirical measurements of language vitality (real world demographic strength) and digitality (online presence) with postcolonial theory and epistemic injustice to explain why linguistic inequality in AI systems is not incidental but structural. Analyzing data across all documented human languages, we identify four categories: Strongholds (33%, high vitality and digitality), Digital Echoes (6%, high digitality despite declining vitality), Fading Voices (36%, low on both dimensions), and critically, Invisible Giants (27%, high vitality but near-zero digitality) - languages spoken by millions yet absent from the LLM universe. We demonstrate that these patterns reflect continuities from colonial-era linguistic hierarchies to contemporary AI development, constituting what we term digital epistemic injustice. Our analysis reveals that English dominance in AI is not a technical necessity but an artifact of power structures that systematically exclude marginalized linguistic knowledge. We conclude with implications for decolonizing language technology and democratizing access to AI benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11556v1">Personalized and Constructive Feedback for Computer Science Students Using the Large Language Model (LLM)</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The evolving pedagogy paradigms are leading toward educational transformations. One fundamental aspect of effective learning is relevant, immediate, and constructive feedback to students. Providing constructive feedback to large cohorts in academia is an ongoing challenge. Therefore, academics are moving towards automated assessment to provide immediate feedback. However, current approaches are often limited in scope, offering simplistic responses that do not provide students with personalized feedback to guide them toward improvements. This paper addresses this limitation by investigating the performance of Large Language Models (LLMs) in processing students assessments with predefined rubrics and marking criteria to generate personalized feedback for in-depth learning. We aim to leverage the power of existing LLMs for Marking Assessments, Tracking, and Evaluation (LLM-MATE) with personalized feedback to enhance students learning. To evaluate the performance of LLM-MATE, we consider the Software Architecture (SA) module as a case study. The LLM-MATE approach can help module leaders overcome assessment challenges with large cohorts. Also, it helps students improve their learning by obtaining personalized feedback in a timely manner. Additionally, the proposed approach will facilitate the establishment of ground truth for automating the generation of students assessment feedback using the ChatGPT API, thereby reducing the overhead associated with large cohort assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13837v3">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 30 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v2">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11498v1">ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel at algorithmic code generation, they struggle with front-end development, where correctness is judged on rendered pixels and interaction. We present ReLook, an agentic, vision-grounded reinforcement learning framework that empowers an agent to close a robust generate--diagnose--refine loop by invoking a multimodal LLM (MLLM) as a tool. During training, the agent uses the MLLM-in-the-loop both as a visual critic--scoring code with screenshots--and as a source of actionable, vision-grounded feedback; a strict zero-reward rule for invalid renders anchors renderability and prevents reward hacking. To prevent behavioral collapse, we introduce Forced Optimization, a strict acceptance rule that admits only improving revisions, yielding monotonically better trajectories. At inference, we decouple the critic and run a lightweight, critic-free self-edit cycle, keeping latency comparable to base decoding while retaining most of the gains. Across three widely used benchmarks, ReLook consistently outperforms strong baselines in vision-grounded front-end code generation, highlighting the benefits of agentic perception, visual rewards, and training-inference decoupling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04641v2">Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we present a comprehensive evaluation framework aimed at English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task using a demographic-focused taxonomy. We then conduct a systematic evaluation with models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable auditing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02502v3">LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 ACL 2025, our code is available at https://github.com/ZNLP/LADM
    </div>
    <details class="paper-abstract">
      Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11457v1">From <Answer> to <Think>: Multidimensional Supervision of Reasoning Process for LLM Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Improving the multi-step reasoning ability of Large Language Models (LLMs) is a critical yet challenging task. The dominant paradigm, outcome-supervised reinforcement learning (RLVR), rewards only correct final answers, often propagating flawed reasoning and suffering from sparse reward signals. While process-level reward models (PRMs) provide denser, step-by-step feedback, they lack generalizability and interpretability, requiring task-specific segmentation of the reasoning process. To this end, we propose the Dimension-level Reward Model (DRM), a new supervision framework that bridges the gap between these two approaches. DRM evaluates the quality of a reasoning process along three fundamental, complementary, and interpretable dimensions: Confidence for uncertainty calibration, Relevance for semantic alignment, and Coherence for logical consistency. Together, these dimensions capture aspects beyond final answer correctness and enable interpretable assessment without requiring ground truth answers. Experimental results show that DRM provides effective supervision signals, guides the optimization of LLMs and enhances their reasoning ability. In particular, DRM-supervised training achieves consistent gains on both in-distribution and out-of-distribution open-domain tasks, including mathematics, question answering, code execution, and puzzles. Our findings demonstrate that multidimensional supervision of the reasoning process can improve the generalized reasoning ability of LLMs beyond the training distribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05605v4">Evolving LLMs' Self-Refinement Capability via Synergistic Training-Inference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Self-Refinement refers to a model's ability to revise its own responses to produce improved outputs. This capability can also serve as a fundamental mechanism for Self-Improvement, for example, by reconstructing datasets with refined results to enhance intrinsic model performance. However, our comprehensive experiments reveal that large language models (LLMs) show no clear evidence of inherent Self-Refinement and may even experience response quality degradation after Self-Refinement. To address this issue, we propose EVOLVE, a simple and effective framework for eliciting and tracking the evolution of Self-Refinement through iterative training. We first explore optimization methods during training to activate the model's Self-Refinement capability. Then, at inference, we investigate various generation strategies to further enhance and utilize Self-Refinement while supplying the necessary data for training. Through synergistic optimization of training and inference stages, we continually evolve the model's Self-Refinement ability, enabling it to better refine its own responses. Moreover, we demonstrate the potential of leveraging Self-Refinement to achieve broader Self-Improvement of intrinsic model abilities. Experiments show that the evolved Self-Refinement ability enables the Llama-3.1-8B base model to surpass GPT-4o, achieving 62.3% length-controlled and 63.3% raw win rates on AlpacaEval 2, and 50.3% on Arena-Hard. It also generalizes effectively to out-of-domain reasoning tasks, improving performance on mathematical reasoning benchmarks such as GSM8K and MATH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11434v1">Who are you, ChatGPT? Personality and Demographic Style in LLM-Generated Content</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 ECAI2025 (Identity-Aware AI workshop)
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have become central to everyday life, producing human-like text across diverse domains. A growing body of research investigates whether these models also exhibit personality- and demographic-like characteristics in their language. In this work, we introduce a novel, data-driven methodology for assessing LLM personality without relying on self-report questionnaires, applying instead automatic personality and gender classifiers to model replies on open-ended questions collected from Reddit. Comparing six widely used models to human-authored responses, we find that LLMs systematically express higher Agreeableness and lower Neuroticism, reflecting cooperative and stable conversational tendencies. Gendered language patterns in model text broadly resemble those of human writers, though with reduced variation, echoing prior findings on automated agents. We contribute a new dataset of human and model responses, along with large-scale comparative analyses, shedding new light on the topic of personality and demographic patterns of generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11423v1">Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), enables users to flag misleading posts, attach contextual notes, and vote on their helpfulness. However, our analysis of 30.8K health-related notes reveals significant latency, with a median delay of 17.6 hours before the first note receives a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified framework that leverages large language models (LLMs) to augment Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two complementary modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, along with a hierarchical three-step evaluation that progressively assesses relevance, correctness, and helpfulness. We instantiate the framework through HealthNotes, a benchmark of 1.2K helpfulness-annotated health notes paired with a fine-tuned helpfulness judge. Experiments on fifteen LLMs reveal an overlooked loophole in current helpfulness evaluation, where stylistic fluency is mistaken for factual accuracy, and demonstrate that our hierarchical evaluation and LLM-augmented generation jointly enhance factual precision and evidence utility. These results point toward a hybrid human-AI governance model that improves both the rigor and timeliness of crowd-sourced fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11414v1">Uncertainty-Aware, Risk-Adaptive Access Control for Agentic Systems using an LLM-Judged TBAC Model</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The proliferation of autonomous AI agents within enterprise environments introduces a critical security challenge: managing access control for emergent, novel tasks for which no predefined policies exist. This paper introduces an advanced security framework that extends the Task-Based Access Control (TBAC) model by using a Large Language Model (LLM) as an autonomous, risk-aware judge. This model makes access control decisions not only based on an agent's intent but also by explicitly considering the inherent \textbf{risk associated with target resources} and the LLM's own \textbf{model uncertainty} in its decision-making process. When an agent proposes a novel task, the LLM judge synthesizes a just-in-time policy while also computing a composite risk score for the task and an uncertainty estimate for its own reasoning. High-risk or high-uncertainty requests trigger more stringent controls, such as requiring human approval. This dual consideration of external risk and internal confidence allows the model to enforce a more robust and adaptive version of the principle of least privilege, paving the way for safer and more trustworthy autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11409v1">Leveraging LLMs for Semi-Automatic Corpus Filtration in Systematic Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The creation of systematic literature reviews (SLR) is critical for analyzing the landscape of a research field and guiding future research directions. However, retrieving and filtering the literature corpus for an SLR is highly time-consuming and requires extensive manual effort, as keyword-based searches in digital libraries often return numerous irrelevant publications. In this work, we propose a pipeline leveraging multiple large language models (LLMs), classifying papers based on descriptive prompts and deciding jointly using a consensus scheme. The entire process is human-supervised and interactively controlled via our open-source visual analytics web interface, LLMSurver, which enables real-time inspection and modification of model outputs. We evaluate our approach using ground-truth data from a recent SLR comprising over 8,000 candidate papers, benchmarking both open and commercial state-of-the-art LLMs from mid-2024 and fall 2025. Results demonstrate that our pipeline significantly reduces manual effort while achieving lower error rates than single human annotators. Furthermore, modern open-source models prove sufficient for this task, making the method accessible and cost-effective. Overall, our work demonstrates how responsible human-AI collaboration can accelerate and enhance systematic literature reviews within academic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11398v1">Living Off the LLM: How LLMs Will Change Adversary Tactics</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 6 pages, 0 figures
    </div>
    <details class="paper-abstract">
      In living off the land attacks, malicious actors use legitimate tools and processes already present on a system to avoid detection. In this paper, we explore how the on-device LLMs of the future will become a security concern as threat actors integrate LLMs into their living off the land attack pipeline and ways the security community may mitigate this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11389v1">Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 34 pages, 32figures
    </div>
    <details class="paper-abstract">
      Social deduction games like Werewolf combine language, reasoning, and strategy, providing a testbed for studying natural language and social intelligence. However, most studies reduce the game to LLM-based self-play, yielding templated utterances and anecdotal cases that overlook the richness of social gameplay. Evaluation further relies on coarse metrics such as survival time or subjective scoring due to the lack of quality reference data. To address these gaps, we curate a high-quality, human-verified multimodal Werewolf dataset containing over 100 hours of video, 32.4M utterance tokens, and 15 rule variants. Based on this dataset, we propose a novel strategy-alignment evaluation that leverages the winning faction's strategies as ground truth in two stages: 1) Speech evaluation, formulated as multiple-choice-style tasks that assess whether the model can adopt appropriate stances across five dimensions of social ability; and 2) Decision evaluation, which assesses the model's voting choices and opponent-role inferences. This framework enables a fine-grained evaluation of models' linguistic and reasoning capabilities, while capturing their ability to generate strategically coherent gameplay. Our experiments show that state-of-the-art LLMs show diverse performance, with roughly half remain below 0.50, revealing clear gaps in deception and counterfactual reasoning. We hope our dataset further inspires research on language, reasoning, and strategy in multi-agent interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25123v2">From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11358v1">LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. While traditional retrieval focuses on relevance, RAG's effectiveness depends on the utility of retrieved passages, i.e., the usefulness in facilitating the generation of an accurate and comprehensive answer. Existing studies often treat utility as a generic attribute, ignoring the fact that different LLMs may benefit differently from the same passage due to variations in internal knowledge and comprehension ability. In this work, we introduce and systematically investigate the notion of LLM-specific utility. Through large-scale experiments across multiple datasets and LLMs, we demonstrate that human-annotated passages are not optimal for LLMs and that ground-truth utilitarian passages are not transferable across different LLMs. These findings highlight the necessity of adopting the LLM-specific utility in RAG research. Our findings indicate that some human-annotated passages are not ground-truth utilitarian passages for specific LLMs, partially due to the varying readability of queries and passages for LLMs, a tendency for which perplexity is a key metric. Based on these findings, we propose a benchmarking procedure for LLM-specific utility judgments. We evaluate existing utility judgment methods on six datasets and find that while verbalized methods using pseudo-answers perform robustly, LLMs struggle to assess utility effectively-failing to reject all passages for known queries and to select truly useful ones for unknown queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00761v3">Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often fragile: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources. In this work, we take a different perspective by investigating the role of the optimizer, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the 'grade' of the optimizer, defined by the level of information it exploits, ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based), is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (e.g., gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning. Motivated by these insights, we propose a hybrid optimizer that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11331v1">Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference at the network edge is a promising serving paradigm that leverages distributed edge resources to run inference near users and enhance privacy. Existing edge-based LLM inference systems typically adopt autoregressive decoding (AD), which only generates one token per forward pass. This iterative process, compounded by the limited computational resources of edge nodes, results in high serving latency and constrains the system's ability to support multiple users under growing demands.To address these challenges, we propose a speculative decoding (SD)-based LLM serving framework that deploys small and large models across heterogeneous edge nodes to collaboratively deliver inference services. Specifically, the small model rapidly generates draft tokens that the large model verifies in parallel, enabling multi-token generation per forward pass and thus reducing serving latency. To improve resource utilization of edge nodes, we incorporate pipeline parallelism to overlap drafting and verification across multiple inference tasks. Based on this framework, we analyze and derive a comprehensive latency model incorporating both communication and inference latency. Then, we formulate a joint optimization problem for speculation length, task batching, and wireless communication resource allocation to minimize total serving latency. To address this problem, we derive the closed-form solutions for wireless communication resource allocation, and develop a dynamic programming algorithm for joint batching and speculation control strategies. Experimental results demonstrate that the proposed framework achieves lower serving latency compared to AD-based serving systems. In addition,the proposed joint optimization method delivers up to 44.9% latency reduction compared to benchmark schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11328v1">Do LLMs "Feel"? Emotion Circuits Discovery and Control</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 19 pages, 8 figures, 8 tables. Code and dataset available at https://github.com/Aurora-cx/EmotionCircuits-LLM
    </div>
    <details class="paper-abstract">
      As the demand for emotional intelligence in large language models (LLMs) grows, a key challenge lies in understanding the internal mechanisms that give rise to emotional expression and in controlling emotions in generated text. This study addresses three core questions: (1) Do LLMs contain context-agnostic mechanisms shaping emotional expression? (2) What form do these mechanisms take? (3) Can they be harnessed for universal emotion control? We first construct a controlled dataset, SEV (Scenario-Event with Valence), to elicit comparable internal states across emotions. Subsequently, we extract context-agnostic emotion directions that reveal consistent, cross-context encoding of emotion (Q1). We identify neurons and attention heads that locally implement emotional computation through analytical decomposition and causal analysis, and validate their causal roles via ablation and enhancement interventions. Next, we quantify each sublayer's causal influence on the model's final emotion representation and integrate the identified local components into coherent global emotion circuits that drive emotional expression (Q2). Directly modulating these circuits achieves 99.65% emotion-expression accuracy on the test set, surpassing prompting- and steering-based methods (Q3). To our knowledge, this is the first systematic study to uncover and validate emotion circuits in LLMs, offering new insights into interpretability and controllable emotional intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11313v1">Automated Skill Decomposition Meets Expert Ontologies: Bridging the Granularity Gap with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      This paper investigates automated skill decomposition using Large Language Models (LLMs) and proposes a rigorous, ontology-grounded evaluation framework. Our framework standardizes the pipeline from prompting and generation to normalization and alignment with ontology nodes. To evaluate outputs, we introduce two metrics: a semantic F1-score that uses optimal embedding-based matching to assess content accuracy, and a hierarchy-aware F1-score that credits structurally correct placements to assess granularity. We conduct experiments on ROME-ESCO-DecompSkill, a curated subset of parents, comparing two prompting strategies: zero-shot and leakage-safe few-shot with exemplars. Across diverse LLMs, zero-shot offers a strong baseline, while few-shot consistently stabilizes phrasing and granularity and improves hierarchy-aware alignment. A latency analysis further shows that exemplar-guided prompts are competitive - and sometimes faster - than unguided zero-shot due to more schema-compliant completions. Together, the framework, benchmark, and metrics provide a reproducible foundation for developing ontology-faithful skill decomposition systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11288v1">Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across three datasets, three frontier models produce broadly misaligned responses at rates between 2% and 17% given 64 narrow in-context examples, and up to 58% with 256 examples. We also examine mechanisms of EM by eliciting step-by-step reasoning (while leaving in-context examples unchanged). Manual analysis of the resulting chain-of-thought shows that 67.5% of misaligned traces explicitly rationalize harmful outputs by adopting a reckless or dangerous ''persona'', echoing prior results on finetuning-induced EM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11282v1">Vision-LLMs for Spatiotemporal Traffic Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Accurate spatiotemporal traffic forecasting is a critical prerequisite for proactive resource management in dense urban mobile networks. While Large Language Models (LLMs) have shown promise in time series analysis, they inherently struggle to model the complex spatial dependencies of grid-based traffic data. Effectively extending LLMs to this domain is challenging, as representing the vast amount of information from dense geographical grids can be inefficient and overwhelm the model's context. To address these challenges, we propose ST-Vision-LLM, a novel framework that reframes spatiotemporal forecasting as a vision-language fusion problem. Our approach leverages a Vision-LLM visual encoder to process historical global traffic matrices as image sequences, providing the model with a comprehensive global view to inform cell-level predictions. To overcome the inefficiency of LLMs in handling numerical data, we introduce an efficient encoding scheme that represents floating-point values as single tokens via a specialized vocabulary, coupled with a two-stage numerical alignment fine-tuning process. The model is first trained with Supervised Fine-Tuning (SFT) and then further optimized for predictive accuracy using Group Relative Policy Optimization (GRPO), a memory-efficient reinforcement learning method. Evaluations on real-world mobile traffic datasets demonstrate that ST-Vision-LLM outperforms existing methods by 15.6% in long-term prediction accuracy and exceeds the second-best baseline by over 30.04% in cross-domain few-shot scenarios. Our extensive experiments validate the model's strong generalization capabilities across various data-scarce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10320v3">J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 10 pages, 13 tables, 14 figures
    </div>
    <details class="paper-abstract">
      The progress of AI is bottlenecked by the quality of evaluation, making powerful LLM-as-a-Judge models a core solution. The efficacy of these judges depends on their chain-of-thought reasoning, creating a critical need for methods that can effectively optimize this reasoning process. In this work, we introduce J1, a reinforcement learning framework for teaching LLM judges to think before making decisions. Our core contribution lies in converting all judgment tasks for non-verifiable and verifiable prompts into a unified format with verifiable rewards, enabling direct optimization of evaluation quality while mitigating positional bias. We then use RL to train thinking-judges at scales of 8B, 32B, and 70B and show that they obtain state-of-the-art performance across multiple benchmarks. In particular, J1-Qwen-32B, our multitasked pointwise and pairwise judge also outperforms o1-mini, o3, and a much larger 671B DeepSeek-R1 on some benchmarks, while only training on synthetic data. Through comprehensive ablations of pairwise, pointwise, and multitask J1 variants, we demonstrate the effectiveness of our approach across seed prompts, reward strategies, and training recipes. Qualitative analysis reveals that J1 develops systematic evaluation strategies, including dynamic criteria generation, reference answer creation, iterative self-correction of initial assessments, and feedback generation for low-quality responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19981v3">Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4\% absolute on SAT MATH). Furthermore, we benchmark our PRM against existing open-source reward models, demonstrating superior alignment with reasoning quality and more consistent guidance for downstream generation. Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11246v1">Collaborative Shadows: Distributed Backdoor Attacks in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) demonstrate increasing integration into next-generation applications, but their safety in backdoor attacks remains largely underexplored. However, existing research has focused exclusively on single-agent backdoor attacks, overlooking the novel attack surfaces introduced by agent collaboration in MAS. To bridge this gap, we present the first Distributed Backdoor Attack tailored to MAS. We decompose the backdoor into multiple distributed attack primitives that are embedded within MAS tools. These primitives remain dormant individually but collectively activate only when agents collaborate in a specific sequence, thereby assembling the full backdoor to execute targeted attacks such as data exfiltration. To fully assess this threat, we introduce a benchmark for multi-role collaborative tasks and a sandboxed framework to evaluate. Extensive experiments demonstrate that our attack achieves an attack success rate exceeding 95% without degrading performance on benign tasks. This work exposes novel backdoor attack surfaces that exploit agent collaboration, underscoring the need to move beyond single-agent protection. Code and benchmark are available at https://github.com/whfeLingYu/Distributed-Backdoor-Attacks-in-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11218v1">The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v2">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11203v1">TraceAegis: Securing LLM-Based Agents via Hierarchical and Behavioral Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      LLM-based agents have demonstrated promising adaptability in real-world applications. However, these agents remain vulnerable to a wide range of attacks, such as tool poisoning and malicious instructions, that compromise their execution flow and can lead to serious consequences like data breaches and financial loss. Existing studies typically attempt to mitigate such anomalies by predefining specific rules and enforcing them at runtime to enhance safety. Yet, designing comprehensive rules is difficult, requiring extensive manual effort and still leaving gaps that result in false negatives. As agent systems evolve into complex software systems, we take inspiration from software system security and propose TraceAegis, a provenance-based analysis framework that leverages agent execution traces to detect potential anomalies. In particular, TraceAegis constructs a hierarchical structure to abstract stable execution units that characterize normal agent behaviors. These units are then summarized into constrained behavioral rules that specify the conditions necessary to complete a task. By validating execution traces against both hierarchical and behavioral constraints, TraceAegis is able to effectively detect abnormal behaviors. To evaluate the effectiveness of TraceAegis, we introduce TraceAegis-Bench, a dataset covering two representative scenarios: healthcare and corporate procurement. Each scenario includes 1,300 benign behaviors and 300 abnormal behaviors, where the anomalies either violate the agent's execution order or break the semantic consistency of its execution sequence. Experimental results demonstrate that TraceAegis achieves strong performance on TraceAegis-Bench, successfully identifying the majority of abnormal behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v4">LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted as a main paper at EMNLP 2025. CR version
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics for literature prioritize mechanical accuracy over artistic expression and tend to overrate machine translation as being superior to human translation from experienced professionals. In the long run, this bias could result in an irreversible decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LITRANSPROQA, a novel, reference-free, LLM-based question-answering framework designed for literary translation evaluation. LITRANSPROQA integrates humans in the loop to incorporate insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LITRANSPROQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LITRANSPROQA reaches an adequacy performance comparable to trained linguistic student evaluators, though it still falls behind experienced professional translators. LITRANSPROQA shows broad applicability to open-source models like LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free tool for evaluating literary translations that require local processing due to copyright or ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11192v1">Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 8 pages, to appear in IEEE Cross-disciplinary Conference on Memory-Centric Computing (CCMCC)
    </div>
    <details class="paper-abstract">
      Structured sparsity enables deploying large language models (LLMs) on resource-constrained systems. Approaches like dense-to-sparse fine-tuning are particularly compelling, achieving remarkable structured sparsity by reducing the model size by over 6.7x, while still maintaining acceptable accuracy. Despite this reduction, LLM inference, especially the decode stage being inherently memory-bound, is extremely expensive on conventional Von-Neumann architectures. Compute-in-memory (CIM) architectures mitigate this by performing computations directly in memory, and when paired with sparse LLMs, enable storing and computing the entire model in memory, eliminating the data movement on the off-chip bus and improving efficiency. Nonetheless, naively mapping sparse matrices onto CIM arrays leads to poor array utilization and diminished computational efficiency. In this paper, we present an automated framework with novel mapping and scheduling strategies to accelerate sparse LLM inference on CIM accelerators. By exploiting block-diagonal sparsity, our approach improves CIM array utilization by over 50%, achieving more than 4x reduction in both memory footprint and the number of required floating-point operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11188v1">Protein as a Second Language for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Main paper: 9 pages, 6 figures. With references and appendix: 18 pages, 9 figures total. Submitted to ICLR 2026 (under review)
    </div>
    <details class="paper-abstract">
      Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the "Protein-as-Second-Language" framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence-question-answer triples that reveal functional cues in a zero-shot setting, without any further training. To support this process, we curate a bilingual corpus of 79,926 protein-QA instances spanning attribute prediction, descriptive understanding, and extended reasoning. Empirically, our method delivers consistent gains across diverse open-source LLMs and GPT-4, achieving up to 17.2% ROUGE-L improvement (average +7%) and even surpassing fine-tuned protein-specific language models. These results highlight that generic LLMs, when guided with protein-as-language cues, can outperform domain-specialized models, offering a scalable pathway for protein understanding in foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04651v2">Agents of Change: Self-Evolving LLM Agents for Strategic Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      We address the long-horizon gap in large language model (LLM) agents by enabling them to sustain coherent strategies in adversarial, stochastic environments. Settlers of Catan provides a challenging benchmark: success depends on balancing short- and long-term goals amid randomness, trading, expansion, and blocking. Prompt-centric LLM agents (e.g., ReAct, Reflexion) must re-interpret large, evolving game states each turn, quickly saturating context windows and losing strategic consistency. We propose HexMachina, a continual learning multi-agent system that separates environment discovery (inducing an adapter layer without documentation) from strategy improvement (evolving a compiled player through code refinement and simulation). This design preserves executable artifacts, allowing the LLM to focus on high-level strategy rather than per-turn reasoning. In controlled Catanatron experiments, HexMachina learns from scratch and evolves players that outperform the strongest human-crafted baseline (AlphaBeta), achieving a 54% win rate and surpassing prompt-driven and no-discovery baselines. Ablations confirm that isolating pure strategy learning improves performance. Overall, artifact-centric continual learning transforms LLMs from brittle stepwise deciders into stable strategy designers, advancing long-horizon autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11151v1">TypePilot: Leveraging the Scala Type System for Secure LLM-generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language Models (LLMs) have shown remarkable proficiency in code generation tasks across various programming languages. However, their outputs often contain subtle but critical vulnerabilities, posing significant risks when deployed in security-sensitive or mission-critical systems. This paper introduces TypePilot, an agentic AI framework designed to enhance the security and robustness of LLM-generated code by leveraging strongly typed and verifiable languages, using Scala as a representative example. We evaluate the effectiveness of our approach in two settings: formal verification with the Stainless framework and general-purpose secure code generation. Our experiments with leading open-source LLMs reveal that while direct code generation often fails to enforce safety constraints, just as naive prompting for more secure code, our type-focused agentic pipeline substantially mitigates input validation and injection vulnerabilities. The results demonstrate the potential of structured, type-guided LLM workflows to improve the SotA of the trustworthiness of automated code generation in high-assurance domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v2">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11129v1">video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Continuous, high-frame-rate, high-resolution processing of long video streams is critical for future AI agents, yet current video-understanding LLMs struggle to scale. Offline, fixed-frame-number methods require the stream length to adapt frame rates; streaming methods constrain memory by merging or discarding tokens, losing information. We propose video-SALMONN S, a streaming audio-visual LLM that, to our knowledge, is the first to process 3-hour videos at 1 FPS and 360p resolution under a fixed memory budget. Our model introduces (i) a test-time-training (TTT) memory module that continually updates token representations to capture long-range dependencies by replacing token merging, and (ii) a prompt-dependent memory reader that selectively retrieves context-relevant content from fixed-size memory. The TTT module is optimised with a Hessian-free conjugate-gradient procedure (TTT_HF) for efficient adaptation. On long-video benchmarks (Video-MME, LVBench, VideoEvalPro), video-SALMONN S sustains high-quality understanding on multi-hour videos with 10k frames and 1M tokens. Our 8B-parameter model achieves 74.2% overall and 67.8% on the Video-MME long split, outperforming both offline and streaming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11121v1">Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used as automated heuristic designers for vehicle routing problems (VRPs), current state-of-the-art methods predominantly rely on prompting massive, general-purpose models like GPT-4. This work challenges that paradigm by demonstrating that a smaller, specialized LLM, when meticulously fine-tuned, can generate components that surpass expert-crafted heuristics within advanced solvers. We propose RFTHGS, a novel Reinforcement learning (RL) framework for Fine-Tuning a small LLM to generate high-performance crossover operators for the Hybrid Genetic Search (HGS) solver, applied to the Capacitated VRP (CVRP). Our method employs a multi-tiered, curriculum-based reward function that progressively guides the LLM to master generating first compilable, then executable, and finally, superior-performing operators that exceed human expert designs. This is coupled with an operator caching mechanism that discourages plagiarism and promotes diversity during training. Comprehensive experiments show that our fine-tuned LLM produces crossover operators which significantly outperform the expert-designed ones in HGS. The performance advantage remains consistent, generalizing from small-scale instances to large-scale problems with up to 1000 nodes. Furthermore, RFTHGS exceeds the performance of leading neuro-combinatorial baselines, prompt-based methods, and commercial LLMs such as GPT-4o and GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11108v1">A Vision for Access Control in LLM-based Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The autonomy and contextual complexity of LLM-based agents render traditional access control (AC) mechanisms insufficient. Static, rule-based systems designed for predictable environments are fundamentally ill-equipped to manage the dynamic information flows inherent in agentic interactions. This position paper argues for a paradigm shift from binary access control to a more sophisticated model of information governance, positing that the core challenge is not merely about permission, but about governing the flow of information. We introduce Agent Access Control (AAC), a novel framework that reframes AC as a dynamic, context-aware process of information flow governance. AAC operates on two core modules: (1) multi-dimensional contextual evaluation, which assesses not just identity but also relationships, scenarios, and norms; and (2) adaptive response formulation, which moves beyond simple allow/deny decisions to shape information through redaction, summarization, and paraphrasing. This vision, powered by a dedicated AC reasoning engine, aims to bridge the gap between human-like nuanced judgment and scalable Al safety, proposing a new conceptual lens for future research in trustworthy agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11104v1">Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Current approaches for strengthening LLM reasoning tend to introduce a training bias toward human-like reasoning trajectories. In step-wise preference optimization, in particular, dependence on human or higher-capacity model annotations for intermediate steps limits exploration of alternative, non-human-like reasoning paths and thus constrains achievable performance. Furthermore, through a small-scale pilot study, we observed that in approximately 75% of cases, the model's first erroneous step occurs after the lowest-confidence point. This suggests that guiding the model at its lowest-confidence point before an error provides more accurate supervision than locating the first explicit error. In this paper, we propose Confidence-Guided Reasoning Path Preference Optimization (CGPO), a method that leverages a confidence signal to identify points of maximal uncertainty in the model's reasoning process and applies self-generated, non-human-like reasoning-path guidance to mitigate trajectory drift. Our experiments span diverse models applied to both code and mathematical reasoning tasks. The results show that, with the same amount of training data, our method using data generated by a small model can achieve better performance in most cases compared with approaches using data generated by a strong model or human-annotated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11076v1">DebugTA: An LLM-Based Agent for Simplifying Debugging and Teaching in Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      In programming education, Debugging and Teaching (DT) task is a common scenario where students receive assistance in correcting their erroneous code. The task involves multiple inputs, including erroneous code, error messages, reference solutions, and the question description, with the goal of generating modification suggestions to the erroneous code. However, two key challenges hinder the effectiveness of existing approaches. Firstly, the complexity and heterogeneity of inputs inherent in DT tasks significantly elevate the reasoning challenges faced by LLMs. Second, existing approaches often fail to fully leverage the availability of standard code in DT tasks, forcing models to rely solely on complex multi-step reasoning, which limits the potential of LLMs in addressing DT tasks effectively. To address these challenges, we propose DebugTA, a novel LLM-based debugging and teaching agent with specialized tools for standard code retrieval, variable substitution to align reference code, and an external compiler for real-time code analysis. Guided by explicit pedagogical and debugging principles, DebugTA acts as an agent that decomposes a complex task into sequential LLM interactions, each utilizing distinct tools for specific subtasks, thereby simplifying the logical reasoning at each step and reducing overall reasoning complexity. Furthermore, DebugTA utilizes tool calls to align the standard code with the erroneous code as much as possible, allowing the LLM to focus on logic errors within the erroneous code and improving the accuracy of the generated suggestions. To rigorously assess the quality of modification suggestions, we introduce a student simulator-teacher interaction paradigm. Experimental results on three real-world code datasets demonstrate that DebugTA consistently improves teaching effectiveness while significantly reducing computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v3">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v4">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Need to address additional data or methodological concerns
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine reasoning or superficial pattern recognition. Common evaluation methods, which focus on the either the final answer or the reasoning process, fail to assess the entire problem-solving procedure. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework, together with its corresponding benchmark, SMART-Bench. SMART decomposes the entire problem solving process into four distinct cognitive dimensions: Understanding, Reasoning, Arithmetic, and Reflection \& Refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings reveal genuine weaknesses in current LLMs and motivate a new metric, the All-Pass Score, to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11062v1">Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11056v1">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11040v1">Enabling Doctor-Centric Medical AI with LLMs through Workflow-Aligned Tasks and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has transformed healthcare by offering clinical guidance, yet their direct deployment to patients poses safety risks due to limited domain expertise. To mitigate this, we propose repositioning LLMs as clinical assistants that collaborate with experienced physicians rather than interacting with patients directly. We conduct a two-stage inspiration-feedback survey to identify real-world needs in clinical workflows. Guided by this, we construct DoctorFLAN, a large-scale Chinese medical dataset comprising 92,000 Q&A instances across 22 clinical tasks and 27 specialties. To evaluate model performance in doctor-facing applications, we introduce DoctorFLAN-test (550 single-turn Q&A items) and DotaBench (74 multi-turn conversations). Experimental results with over ten popular LLMs demonstrate that DoctorFLAN notably improves the performance of open-source LLMs in medical contexts, facilitating their alignment with physician workflows and complementing existing patient-oriented models. This work contributes a valuable resource and framework for advancing doctor-centered medical LLM development
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12575v2">DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at https://github.com/whfeLingYu/DemonAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05431v2">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework specifically tailored for patent classification, which treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset, a widely used benchmark for patent classification, show that our method outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25279v2">RL in the Wild: Characterizing RLVR Training in LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 20 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now widely used across many domains. With their rapid development, Reinforcement Learning with Verifiable Rewards (RLVR) has surged in recent months to enhance their reasoning and understanding abilities. However, its complex data flows and diverse tasks pose substantial challenges to RL training systems, and there is limited understanding of RLVR from a system perspective. To thoroughly understand the system challenges introduced by RLVR, we present a characterization study of RLVR tasks in our LLM deployment. Specifically, we investigate the distribution and variation trends of workloads across different RL tasks across training steps. We identify issues such as GPU idling caused by skewed sequence length distribution, inefficient parallel strategies in dynamically varying workloads, inefficient data management mechanisms, and load imbalance. We describe our observations and call for further investigation into the remaining open challenges. Furthermore, we propose PolyTrace benchmark suite to conduct evaluation with realistic workloads, and a practical use case validates that PolyTrace benchmark suite exhibits 94.7% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10998v1">ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 28 pages, 11 figures, 16 tables. In submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were further amplified by sharp increases in intersectional harms (e.g., Tokenism) for gender and caste-marginalized disabled candidates, highlighting critical blind spots in current safety tools and the need for intersectional safety evaluations of frontier models in high-stakes domains like hiring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10987v1">DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 14 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at https://github.com/hsannn/ditto.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10978v1">Does LLM Focus on the Right Words? Diagnosing Language Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Language Bias, whereby the model over-relies on auxiliary tokens-such as task descriptions and prefix-generated tokens-while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating language bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates language bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04492v2">Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The advancement of LLMs and their accessibility have triggered renewed interest in multi-agent reinforcement learning as robust and adaptive frameworks for dynamically changing environments. This paper introduces RL-Focal, a two-stage RL agent framework that routes and ensembles LLMs. First, we develop the Decider RL-agent, which learns to dynamically select an ensemble of small size ($m_i$) among $N$ LLMs ($m_i \ll N$) for incoming queries from a user-defined downstream task $i$, by maximizing both error-diversity and reasoning-performance of the selected ensemble through iterative updates of task-adaptive rewards and policy. Second, to enable effective fusion of dynamically selected LLMs, we develop the stage-2 Fusion RL-agent, which learns to resolve reasoning conflicts from different LLMs and dynamically adapts to different ensemble teams composed by the Decider Agent for different downstream tasks. Third, we introduce the focal diversity metric to better model the error correlations among multiple LLMs, further improving the generalization performance of the Decider Agent, which actively prunes the ensemble combinations. By focal diversity, we enhance performance across tasks by effectively promoting reward-aware and policy-adaptive ensemble selection and inference fusion. Extensive evaluations on five benchmarks show that RL-Focal achieves the performance improvement of 8.48\% with an ensemble of small size compared to the best individual LLM in a pool and offers stronger robustness. Code is available at https://github.com/sftekin/rl-focal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10959v1">Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10955v1">HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a surge of research on leveraging large language models (LLMs) for sequential recommendation. LLMs have demonstrated remarkable potential in inferring users' nuanced preferences through fine-grained semantic reasoning. However, they also exhibit a notable limitation in effectively modeling collaborative signals, i.e., behavioral correlations inherent in users' historical interactions. Our empirical analysis further reveals that the attention mechanisms in LLMs tend to disproportionately focus on tokens within the same item, thereby impeding the capture of cross-item correlations. To address this limitation, we propose a novel hierarchical attention masking strategy for LLM-based recommendation, termed HatLLM. Specifically, in shallow layers, HatLLM masks attention between tokens from different items, facilitating intra-item semantic understanding; in contrast, in deep layers, HatLLM masks attention within items, thereby compelling the model to capture cross-item correlations. This progressive, layer-wise approach enables LLMs to jointly model both token-level and item-level dependencies. Extensive experiments on three real-world datasets demonstrate that HatLLM achieves significant performance gains (9.13% on average) over existing LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09393v2">ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender systems. Existing approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity users. To address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity users. With a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce platform. ChoirRec improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08948v2">SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from JD dot com demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on JD dot com has substantially enhanced the efficiency of case investigation workflows for risk managers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v4">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15507v5">Steering LLMs for Formal Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent advances in automated theorem proving use Large Language Models (LLMs) to translate informal mathematical statements into formal proofs. However, informal cues are often ambiguous or lack strict logical structure, making it hard for models to interpret them precisely. While existing methods achieve strong performance, little is known about how LLMs internally represent informal cues, or how these influence proof generation. To address this, we explore \textit{activation steering}, an inference-time intervention that identifies linear directions in residual activations associated with informal reasoning traces and adjusts them to improve proof construction without fine-tuning. This mechanism also yields interpretable information about how reasoning is internally encoded in the activation space of LLMs. We test our method for generating formal proofs from already-formalized theorems. Our contributions are twofold: (1) a novel activation-based intervention for guiding proof synthesis in LLMs; and (2) demonstration that this intervention improves performance under two decoding strategies (sampling and best-first search) without any further training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10895v1">LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 This work has been submitted to IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Medium Access Control (MAC) protocols, essential for wireless networks, are typically manually configured. While deep reinforcement learning (DRL)-based protocols enhance task-specified network performance, they suffer from poor generalizability and resilience, demanding costly retraining to adapt to dynamic environments. To overcome this limitation, we introduce a game-theoretic LLM-empowered multi-agent DRL (MARL) framework, in which the uplink transmission between a base station and a varying number of user equipments is modeled as a dynamic multi-follower Stackelberg game (MFSG), capturing the network's natural hierarchical structure. Within this game, LLM-driven agents, coordinated through proximal policy optimization (PPO), synthesize adaptive, semantic MAC protocols in response to network dynamics. Protocol action grammar (PAG) is employed to ensure the reliability and efficiency of this process. Under this system, we further analyze the existence and convergence behavior in terms of a Stackelberg equilibrium by studying the learning dynamics of LLM-empowered unified policies in response to changing followers. Simulations corroborate that our framework achieves a 77.6% greater throughput and a 65.2% fairness improvement over conventional baselines. Besides, our framework generalizes excellently to a fluctuating number of users without requiring retraining or architectural changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07680v3">Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue, but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MoE model, our method speeds up the training by 2.4$\times$ with competitive performance. Codes will be released at https://github.com/ModelTC/HBP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10890v1">LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted by EMNLP2025 System Demonstration
    </div>
    <details class="paper-abstract">
      We introduce LLM x MapReduce-V3, a hierarchically modular agent system designed for long-form survey generation. Building on the prior work, LLM x MapReduce-V2, this version incorporates a multi-agent architecture where individual functional components, such as skeleton initialization, digest construction, and skeleton refinement, are implemented as independent model-context-protocol (MCP) servers. These atomic servers can be aggregated into higher-level servers, creating a hierarchically structured system. A high-level planner agent dynamically orchestrates the workflow by selecting appropriate modules based on their MCP tool descriptions and the execution history. This modular decomposition facilitates human-in-the-loop intervention, affording users greater control and customization over the research process. Through a multi-turn interaction, the system precisely captures the intended research perspectives to generate a comprehensive skeleton, which is then developed into an in-depth survey. Human evaluations demonstrate that our system surpasses representative baselines in both content depth and length, highlighting the strength of MCP-based modular planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16783v3">SubData: Bridging Heterogeneous Datasets to Enable Theory-Driven Evaluation of Political and Demographic Perspectives in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 14 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      As increasingly capable large language models (LLMs) emerge, researchers have begun exploring their potential for subjective tasks. While recent work demonstrates that LLMs can be aligned with diverse human perspectives, evaluating this alignment on downstream tasks (e.g., hate speech detection) remains challenging due to the use of inconsistent datasets across studies. To address this issue, in this resource paper we propose a two-step framework: we (1) introduce SubData, an open-source Python library designed for standardizing heterogeneous datasets to evaluate LLMs perspective alignment; and (2) present a theory-driven approach leveraging this library to test how differently-aligned LLMs (e.g., aligned with different political viewpoints) classify content targeting specific demographics. SubData's flexible mapping and taxonomy enable customization for diverse research needs, distinguishing it from existing resources. We illustrate its usage with an example application and invite contributions to extend our initial release into a multi-construct benchmark suite for evaluating LLMs perspective alignment on natural language processing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10849v1">Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Learning on text-attributed graphs has motivated the use of Large Language Models (LLMs) for graph learning. However, most fusion strategies are applied uniformly across all nodes and attain only small overall performance gains. We argue this result stems from aggregate metrics that obscure when LLMs provide benefit, inhibiting actionable signals for new strategies. In this work, we reframe LLM-GNN fusion around nodes where GNNs typically falter. We first show that performance can significantly differ between GNNs and LLMs, with each excelling on distinct structural patterns, such as local homophily. To leverage this finding, we propose GLANCE (GNN with LLM Assistance for Neighbor- and Context-aware Embeddings), a framework that invokes an LLM to refine a GNN's prediction. GLANCE employs a lightweight router that, given inexpensive per-node signals, decides whether to query the LLM. Since the LLM calls are non-differentiable, the router is trained with an advantage-based objective that compares the utility of querying the LLM against relying solely on the GNN. Across multiple benchmarks, GLANCE achieves the best performance balance across node subgroups, achieving significant gains on heterophilous nodes (up to $+13\%$) while simultaneously achieving top overall performance. Our findings highlight the value of adaptive, node-aware GNN-LLM architectures, where selectively invoking the LLM enables scalable deployment on large graphs without incurring high computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19073v4">MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Multi-hop Hate Speech Explanation</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Jackson Trager and Francielle Vargas contributed equally
    </div>
    <details class="paper-abstract">
      Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via multi-hop hate speech explanation using the Moral Foundations Theory. MFTCXplain comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Our results show a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. Our findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10813v1">LLMs as Strategic Agents: Beliefs, Best Response Behavior, and Emergent Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to domains that require reasoning about other agents' behavior, such as negotiation, policy design, and market simulation, yet existing research has mostly evaluated their adherence to equilibrium play or their exhibited depth of reasoning. Whether they display genuine strategic thinking, understood as the coherent formation of beliefs about other agents, evaluation of possible actions, and choice based on those beliefs, remains unexplored. We develop a framework to identify this ability by disentangling beliefs, evaluation, and choice in static, complete-information games, and apply it across a series of non-cooperative environments. By jointly analyzing models' revealed choices and reasoning traces, and introducing a new context-free game to rule out imitation from memorization, we show that current frontier models exhibit belief-coherent best-response behavior at targeted reasoning depths. When unconstrained, they self-limit their depth of reasoning and form differentiated conjectures about human and synthetic opponents, revealing an emergent form of meta-reasoning. Under increasing complexity, explicit recursion gives way to internally generated heuristic rules of choice that are stable, model-specific, and distinct from known human biases. These findings indicate that belief coherence, meta-reasoning, and novel heuristic formation can emerge jointly from language modeling objectives, providing a structured basis for the study of strategic cognition in artificial agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10806v1">Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Waiting for Conference Response
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are adept at generating responses based on information within their context. While this ability is useful for interacting with structured data like code files, another popular method, Retrieval-Augmented Generation (RAG), retrieves relevant documents to augment the model's in-context learning. However, it is not well-explored how to best represent this retrieved knowledge for generating responses on structured data, particularly hierarchical structures like trees. In this work, we propose a novel bottom-up method to linearize knowledge from tree-like structures (like a GitHub repository) by generating implicit, aggregated summaries at each hierarchical level. This approach enables the knowledge to be stored in a knowledge base and used directly with RAG. We then compare our method to using RAG on raw, unstructured code, evaluating the accuracy and quality of the generated responses. Our results show that while response quality is comparable across both methods, our approach generates over 68% fewer documents in the retriever, a significant gain in efficiency. This finding suggests that leveraging implicit, linearized knowledge may be a highly effective and scalable strategy for handling complex, hierarchical data structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08093v2">Culturally transmitted color categories in LLMs reflect a learning bias toward efficient compression</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted at CogInterp: Interpreting Cognition in Deep Learning Models Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Converging evidence suggests that systems of semantic categories across human languages achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy principle. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-like semantic systems? To address this question, we focus on the domain of color as a key testbed of cognitive theories of categorization and replicate with LLMs (Gemini 2.0-flash and Llama 3.3-70B-Instruct) two influential human behavioral studies. First, we conduct an English color-naming study, showing that Gemini aligns well with the naming patterns of native English speakers and achieves a significantly high IB-efficiency score, while Llama exhibits an efficient but lower complexity system compared to English. Second, to test whether LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via iterated in-context language learning. We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency and increased alignment with patterns observed across the world's languages. These findings demonstrate that LLMs are capable of evolving perceptually grounded, human-like semantic systems, driven by the same fundamental principle that governs semantic efficiency across human languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10739v1">A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions: Dynamical Systems Analysis with Code Generation Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Peer-reviewed and accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) DynaFront 2025 Workshop (https://sites.google.com/view/dynafrontneurips25)
    </div>
    <details class="paper-abstract">
      We introduce a general stochastic differential equation framework for modelling multiobjective optimization dynamics in iterative Large Language Model (LLM) interactions. Our framework captures the inherent stochasticity of LLM responses through explicit diffusion terms and reveals systematic interference patterns between competing objectives via an interference matrix formulation. We validate our theoretical framework using iterative code generation as a proof-of-concept application, analyzing 400 sessions across security, efficiency, and functionality objectives. Our results demonstrate strategy-dependent convergence behaviors with rates ranging from 0.33 to 1.29, and predictive accuracy achieving R2 = 0.74 for balanced approaches. This work proposes the feasibility of dynamical systems analysis for multi-objective LLM interactions, with code generation serving as an initial validation domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04064v2">Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 10 pages, 7 figures, 4 tables. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to navigate the nuances of human emotion. While research confirms that LLMs can simulate emotional intelligence, their internal emotional mechanisms remain largely unexplored. This paper investigates the latent emotional representations within modern LLMs by asking: how, where, and for how long is emotion encoded in their neural architecture? To address this, we introduce a novel, large-scale Reddit corpus of approximately 400,000 utterances, balanced across seven basic emotions through a multi-stage process of classification, rewriting, and synthetic generation. Using this dataset, we employ lightweight "probes" to read out information from the hidden layers of various Qwen3 and LLaMA models without altering their parameters. Our findings reveal that LLMs develop a surprisingly well-defined internal geometry of emotion, which sharpens with model scale and significantly outperforms zero-shot prompting. We demonstrate that this emotional signal is not a final-layer phenomenon but emerges early and peaks mid-network. Furthermore, the internal states are both malleable (they can be influenced by simple system prompts) and persistent, as the initial emotional tone remains detectable for hundreds of subsequent tokens. We contribute our dataset, an open-source probing toolkit, and a detailed map of the emotional landscape within LLMs, offering crucial insights for developing more transparent and aligned AI systems. The code and dataset are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10703v1">Adaptive Selection of Symbolic Languages for Improving LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) still struggle with complex logical reasoning. While previous works achieve remarkable improvements, their performance is highly dependent on the correctness of translating natural language (NL) problems into a symbolic language (SL). Though numerous works focusing on improving this translation accuracy, they only consider the similarity between the meaning of SL and NL, overlooking another crucial influencing factor, the selection of the target SL type itself. For example, first-order logic language specializes in logical reasoning with categorical syllogisms and complex quantifiers, while Boolean satisfiability formalism excels at representing constraint satisfaction like partial problems. To our knowledge, this is the first paper to claim and verify that different NL logical reasoning problem corresponds to different optimal SL formalization for translation. Based on this, we propose a methods to improve the logical reasoning performance of LLMs by adaptively selecting the most suitable SL for each problem prior to translation. Specifically, we leverage LLMs to select the target SL among first-order logic, logic programming and Boolean satisfiability and then translate the problem in NL to target SL expressions as well as employ the corresponding logical solver to derive the final answer. Experimental results on benchmarks show that our adaptive selection method significantly outperforms translating all into single SL and randomly selecting the SL. On a mixed dataset of these benchmarks, our approach achieves 96% accuracy, which improving performance by 25% compared to the second highest accuracy from the first-order logic translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10677v1">Unlocking LLM Safeguards for Low-Resource Languages via Reasoning and Alignment with Minimal Training Data</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted to MRL Workshop at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have enhanced AI capabilities, but also increased the risk posed by malicious requests, highlighting the need for effective LLM safeguards to detect such queries. Existing approaches largely rely on classifier-based methods that lack interpretability and perform poorly on low-resource languages. To address these limitations, we propose ConsistentGuard, a novel reasoning-based multilingual safeguard, which enhances explainability via reasoning and boosts knowledge transfer between languages through alignment. With only 1,000 training samples, our method demonstrates superior performance on three datasets across six languages, outperforming larger models trained with significantly more data, and exhibits strong interpretability and generalization ability. We also contribute a multilingual benchmark extension and release our codes to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18080v2">Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted at NeurIPS 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with the teacher model QwQ-32B-Preview that produces the seed data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10644v1">Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Online ride-hailing platforms aim to deliver efficient mobility-on-demand services, often facing challenges in balancing dynamic and spatially heterogeneous supply and demand. Existing methods typically fall into two categories: reinforcement learning (RL) approaches, which suffer from data inefficiency, oversimplified modeling of real-world dynamics, and difficulty enforcing operational constraints; or decomposed online optimization methods, which rely on manually designed high-level objectives that lack awareness of low-level routing dynamics. To address this issue, we propose a novel hybrid framework that integrates large language model (LLM) with mathematical optimization in a dynamic hierarchical system: (1) it is training-free, removing the need for large-scale interaction data as in RL, and (2) it leverages LLM to bridge cognitive limitations caused by problem decomposition by adaptively generating high-level objectives. Within this framework, LLM serves as a meta-optimizer, producing semantic heuristics that guide a low-level optimizer responsible for constraint enforcement and real-time decision execution. These heuristics are refined through a closed-loop evolutionary process, driven by harmony search, which iteratively adapts the LLM prompts based on feasibility and performance feedback from the optimization layer. Extensive experiments based on scenarios derived from both the New York and Chicago taxi datasets demonstrate the effectiveness of our approach, achieving an average improvement of 16% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10618v1">Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Post-training compression has been a widely employed approach to scale down large language model (LLM) and facilitate efficient inference. In various proposed compression methods, including pruning and quantization, calibration data plays a vital role by informing the weight importance and activation dynamic ranges. However, how calibration data impacts the LLM capability after compression is less explored. Few of the existing works, though recognizing the significance of this study, only investigate the language modeling or commonsense reasoning performance degradation from limited angles, like the data sources or sample amounts. More systematic research is still needed to examine the impacts on different LLM capabilities in terms of compositional properties and domain correspondence of calibration data. In this work, we aim at bridging this gap and further analyze underlying influencing mechanisms from the activation pattern perspective. Especially, we explore the calibration data's impacts on high-level complex reasoning capabilities, like math problem solving and code generation. Delving into the underlying mechanism, we find that the representativeness and diversity in activation space more fundamentally determine the quality of calibration data. Finally, we propose a calibration data curation framework based on such observations and analysis, enhancing the performance of existing post-training compression methods on preserving critical LLM capabilities. Our code is provided in \href{https://github.com/BokwaiHo/COLA.git}{Link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15607v2">From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted to EMNLP 2025 Main as an oral presentation. David Dinucu-Jianu and Jakub Macina contributed equally. Code available: https://github.com/eth-lre/PedagogicalRL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12421v2">Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted by NeurIPS 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Unlike reasoning, which often entails a deep sequence of deductive steps, complex real-world planning is characterized by the need to synthesize a broad spectrum of parallel and potentially conflicting information and constraints. For example, in travel planning scenarios, it requires the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints, leading to suboptimal solutions. Motivated by the challenges of real-world travel planning, this paper introduces the Multiple Aspects of Planning (MAoP), empowering LLMs with "wide-horizon thinking" to solve planning problems with multifaceted constraints. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planners, enabling strong inference-time scalability by scaling aspects to consider various constraints. In addition, existing benchmarks for multi-constraint planning are flawed because they assess constraints in isolation, ignoring causal dependencies within the constraints, e.g, travel planning, where past activities dictate future itinerary. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world simulation, thereby inherently resolving these causal dependencies. This paper advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10592v1">A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Existing studies have introduced method-based reasoning and scope extension as approaches to enhance Large Language Model (LLM) performance beyond direct matrix mappings. Building on these foundations, this paper summarizes and integrates these ideas into a unified Intuition-Method Layered Model with Scope Extension, designed to address indirected (unseen) issues more systematically. In this framework, intuition-based thinking provides rapid first-reaction answers, while method-based thinking decouples questions and solutions into transferable reasoning units. Scope extension is then applied to broaden applicability, including vertical (cause analysis), horizontal (parallel and generalized issues), and for the first time, temporal and spatial extensions, which expand reasoning across time and contextual dimensions. These extensions are organized into systematic knowledge trees that interconnect into a knowledge network, thereby increasing adaptability. To quantitatively evaluate this process, we propose the entropy of method extension, which measures the independence and diversity of extensions as an indicator of the system's capacity to solve unseen questions. By logically connecting existing approaches with new extensions and introducing an entropy-based evaluation framework, this work advances toward a more robust and extensible reasoning paradigm for LLMs in real-world problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10539v1">Detecting Hallucinations in Authentic LLM-Human Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied in sensitive domains such as medicine and law, hallucination detection has become a critical task. Although numerous benchmarks have been proposed to advance research in this area, most of them are artificially constructed--either through deliberate hallucination induction or simulated interactions--rather than derived from genuine LLM-human dialogues. Consequently, these benchmarks fail to fully capture the characteristics of hallucinations that occur in real-world usage. To address this limitation, we introduce AuthenHallu, the first hallucination detection benchmark built entirely from authentic LLM-human interactions. For AuthenHallu, we select and annotate samples from genuine LLM-human dialogues, thereby providing a faithful reflection of how LLMs hallucinate in everyday user interactions. Statistical analysis shows that hallucinations occur in 31.4% of the query-response pairs in our benchmark, and this proportion increases dramatically to 60.0% in challenging domains such as Math & Number Problems. Furthermore, we explore the potential of using vanilla LLMs themselves as hallucination detectors and find that, despite some promise, their current performance remains insufficient in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19657v5">LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted as International Conference of Computational Optimization 2025 Oral
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size and deployment scale, quantization has become an essential technique for reducing memory footprint and improving inference efficiency. However, existing quantization toolkits often lack transparency, flexibility, and system-level scalability across GPUs and distributed environments. We present \textbf{LLMEasyQuant}, a modular, system-aware quantization framework designed for efficient, low-bit inference of LLMs on single-node multi-GPU, multi-node, and edge hardware. LLMEasyQuant supports a wide range of quantization methods -- including Symmetric Quantization, ZeroQuant, SmoothQuant, and SimQuant -- with unified interfaces for per-layer calibration, bitwidth assignment, and runtime adaptation. It integrates fused CUDA kernels with NCCL-based distributed synchronization and supports both static and online quantization. Empirical results show that LLMEasyQuant can achieve substantial speedup in GEMM execution, HBM load time, and near-linear multi-GPU scaling. Ablation studies further validate its ability to balance latency, memory, and accuracy under diverse deployment conditions. LLMEasyQuant offers a practical quantization serving system for scalable, hardware-optimized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11166v2">SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Despite advances in pretraining with extended context lengths, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named $\textbf{S}$h$\textbf{o}$rt-to-$\textbf{Lo}$ng $\textbf{P}$reference $\textbf{O}$ptimization ($\textbf{SoLoPO}$), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency utilization for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24869v2">Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      With the growing popularity of LLM agents and RAG, it has become increasingly important to retrieve documents that are essential for solving a task, even when their connection to the task is indirect or implicit. Addressing this problem requires fine-grained reasoning to accurately assess the relevance between the task and each candidate document. This capability, however, poses a significant challenge for existing IR techniques. Despite recent progress in reasoning-enhanced IR, existing approaches still face significant challenges in applicability, scalability, and efficiency. In this work, we propose Retro*, a novel approach for reasoning-intensive document retrieval. Our method introduces a rubric-based relevance scoring mechanism, enabling the model to reason about the relationship between a task and a document based on explicitly defined criteria, whereby producing a fine-grained, interpretable relevance score. Retro* also supports test-time scaling by combining multiple reasoning trajectories via score integration, which produces more reliable relevance estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel reinforcement learning algorithm tailored for its relevance scoring mechanism, which employs two composite rewards to fully exploit the trajectories of each training sample. Our experiments show that Retro* outperforms existing document retrieval methods with notable advantages, leading to state-of-the-art performance on the BRIGHT benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10517v1">ECO: Enhanced Code Optimization via Performance-Aware Prompting for Code-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Code runtime optimization-the task of rewriting a given code to a faster one-remains challenging, as it requires reasoning about performance trade-offs involving algorithmic and structural choices. Recent approaches employ code-LLMs with slow-fast code pairs provided as optimization guidance, but such pair-based methods obscure the causal factors of performance gains and often lead to superficial pattern imitation rather than genuine performance reasoning. We introduce ECO, a performance-aware prompting framework for code optimization. ECO first distills runtime optimization instructions (ROIs) from reference slow-fast code pairs; Each ROI describes root causes of inefficiency and the rationales that drive performance improvements. For a given input code, ECO in parallel employs (i) a symbolic advisor to produce a bottleneck diagnosis tailored to the code, and (ii) an ROI retriever to return related ROIs. These two outputs are then composed into a performance-aware prompt, providing actionable guidance for code-LLMs. ECO's prompts are model-agnostic, require no fine-tuning, and can be easily prepended to any code-LLM prompt. Our empirical studies highlight that ECO prompting significantly improves code-LLMs' ability to generate efficient code, achieving speedups of up to 7.81x while minimizing correctness loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10605v2">RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07731v2">oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Organic reaction mechanisms are the stepwise elementary reactions by which reactants form intermediates and products, and are fundamental to understanding chemical reactivity and designing new molecules and reactions. Although large language models (LLMs) have shown promise in understanding chemical tasks such as synthesis design, it is unclear to what extent this reflects genuine chemical reasoning capabilities, i.e., the ability to generate valid intermediates, maintain chemical consistency, and follow logically coherent multi-step pathways. We address this by introducing oMeBench, the first large-scale, expert-curated benchmark for organic mechanism reasoning in organic chemistry. It comprises over 10,000 annotated mechanistic steps with intermediates, type labels, and difficulty ratings. Furthermore, to evaluate LLM capability more precisely and enable fine-grained scoring, we propose oMeS, a dynamic evaluation framework that combines step-level logic and chemical similarity. We analyze the performance of state-of-the-art LLMs, and our results show that although current models display promising chemical intuition, they struggle with correct and consistent multi-step reasoning. Notably, we find that using prompting strategy and fine-tuning a specialist model on our proposed dataset increases performance by 50% over the leading closed-source model. We hope that oMeBench will serve as a rigorous foundation for advancing AI systems toward genuine chemical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10493v1">The Hidden DNA of LLM-Generated JavaScript: Structural Patterns Enable High-Accuracy Authorship Attribution</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      In this paper, we present the first large-scale study exploring whether JavaScript code generated by Large Language Models (LLMs) can reveal which model produced it, enabling reliable authorship attribution and model fingerprinting. With the rapid rise of AI-generated code, attribution is playing a critical role in detecting vulnerabilities, flagging malicious content, and ensuring accountability. While AI-vs-human detection usually treats AI as a single category we show that individual LLMs leave unique stylistic signatures, even among models belonging to the same family or parameter size. To this end, we introduce LLM-NodeJS, a dataset of 50,000 Node.js back-end programs from 20 large language models. Each has four transformed variants, yielding 250,000 unique JavaScript samples and two additional representations (JSIR and AST) for diverse research applications. Using this dataset, we benchmark traditional machine learning classifiers against fine-tuned Transformer encoders and introduce CodeT5-JSA, a custom architecture derived from the 770M-parameter CodeT5 model with its decoder removed and a modified classification head. It achieves 95.8% accuracy on five-class attribution, 94.6% on ten-class, and 88.5% on twenty-class tasks, surpassing other tested models such as BERT, CodeBERT, and Longformer. We demonstrate that classifiers capture deeper stylistic regularities in program dataflow and structure, rather than relying on surface-level features. As a result, attribution remains effective even after mangling, comment removal, and heavy code transformations. To support open science and reproducibility, we release the LLM-NodeJS dataset, Google Colab training scripts, and all related materials on GitHub: https://github.com/LLM-NodeJS-dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10486v1">SASER: Stego attacks on open-source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Open-source large language models (LLMs) have demonstrated considerable dominance over proprietary LLMs in resolving neural processing tasks, thanks to the collaborative and sharing nature. Although full access to source codes, model parameters, and training data lays the groundwork for transparency, we argue that such a full-access manner is vulnerable to stego attacks, and their ill-effects are not fully understood. In this paper, we conduct a systematic formalization for stego attacks on open-source LLMs by enumerating all possible threat models associated with adversary objectives, knowledge, and capabilities. Therein, the threat posed by adversaries with internal knowledge, who inject payloads and triggers during the model sharing phase, is of practical interest. We go even further and propose the first stego attack on open-source LLMs, dubbed SASER, which wields impacts through identifying targeted parameters, embedding payloads, injecting triggers, and executing payloads sequentially. Particularly, SASER enhances the attack robustness against quantization-based local deployment by de-quantizing the embedded payloads. In addition, to achieve stealthiness, SASER devises the performance-aware importance metric to identify targeted parameters with the least degradation of model performance. Extensive experiments on LlaMA2-7B and ChatGLM3-6B, without quantization, show that the stealth rate of SASER outperforms existing stego attacks (for general DNNs) by up to 98.1%, while achieving the same attack success rate (ASR) of 100%. More importantly, SASER improves ASR on quantized models from 0 to 100% in all settings. We appeal for investigations on countermeasures against SASER in view of the significant attack effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10467v1">AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) is increasingly constrained by memory and latency bottlenecks, motivating the need for quantization techniques that flexibly balance accuracy and efficiency. Recent work has introduced multi-precision models, which enable inference at multiple precisions within a single model depending on runtime constraints. To support such flexibility, quantized weights are often stored as bit-planes, where hardware efficiency improves when the compute operates directly at the bit-plane level and activates only the precision required by each request. In this work, we present AnyBCQ, a hardware-friendly multi-precision extension of Binary-Coded Quantization (BCQ) that supports direct bit-plane operations. By representing weights as binary bit-planes with corresponding scale factors, AnyBCQ enables bit-plane-level computation and maps naturally to accelerator-friendly, bit-parallel arithmetic. Our progressive precision expansion mechanism incrementally refines scaling factors while reusing previously assigned binary codes, yielding monotonic improvements in accuracy as additional bits are enabled. We further co-design a specialized kernel that exploits the BCQ structure to support dynamic per-request precision selection with negligible overhead. Experiments on recent LLMs demonstrate that AnyBCQ significantly narrows the accuracy drop in the low-bit regime (e.g. 2-bit), remains competitive at higher precision, and achieves throughput gains of up to 3.0x over half precision and 1.2x over state-of-the-art multi-precision methods. By aligning algorithmic flexibility with hardware efficiency, AnyBCQ provides a practical foundation for multi-precision LLM deployment across diverse service-level objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00482v2">Talk Less, Call Right: Enhancing Role-Play LLM Agents with Automatic Prompt Optimization and Role Prompting</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 EMNLP 2025 Wordplay Workshop (Spotlight)
    </div>
    <details class="paper-abstract">
      This report investigates approaches for prompting a tool-augmented large language model (LLM) to act as a role-playing dialogue agent in the API track of the Commonsense Persona-grounded Dialogue Challenge (CPDC) 2025. In this setting, dialogue agents often produce overly long in-character responses (over-speaking) while failing to use tools effectively according to the persona (under-acting), such as generating function calls that do not exist or making unnecessary tool calls before answering. We explore four prompting approaches to address these issues: 1) basic role prompting, 2) improved role prompting, 3) automatic prompt optimization (APO), and 4) rule-based role prompting. The rule-based role prompting (RRP) approach achieved the best performance through two novel techniques-character-card/scene-contract design and strict enforcement of function calling-which led to an overall score of 0.571, improving on the zero-shot baseline score of 0.519. These findings demonstrate that RRP design can substantially improve the effectiveness and reliability of role-playing dialogue agents compared with more elaborate methods such as APO. To support future efforts in developing persona prompts, we are open-sourcing all of our best-performing prompts and the APO tool Source code is available at https://github.com/scb-10x/apo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00882v3">VulSolver: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, limitations in context length, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 96.29% accuracy, 96.55% F1-score, and 100% recall. Applied to popular GitHub repositories, VULSOLVER also identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19274v3">Neuralink: Fast LLM Inference on Smartphones with Neuron Co-Activation Linking</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across various domains, yet deploying them on mobile devices remains an arduous challenge due to their extensive computational and memory demands. While lightweight LLMs have been developed to fit mobile environments, they suffer from degraded model accuracy. In contrast, sparsity-based techniques minimize DRAM usage by selectively transferring only relevant neurons to DRAM while retaining the full model in external storage, such as flash. However, such approaches are critically limited by numerous I/O operations, particularly on smartphones with severe IOPS constraints. In this paper, we propose Neuralink, a novel approach that accelerates LLM inference on smartphones by optimizing neuron placement in flash memory. Neuralink leverages the concept of Neuron Co-Activation, where neurons frequently activated together are linked to facilitate continuous read access and optimize I/O efficiency. Our approach incorporates a two-stage solution: an offline stage that reorganizes neuron placement based on co-activation patterns, and an online stage that employs tailored data access and caching strategies to align well with hardware characteristics. Evaluations conducted on a variety of smartphones and LLMs demonstrate that Neuralink achieves on average $1.49\times$ improvements in end-to-end latency compared to the state-of-the-art. As the first solution to optimize storage placement under sparsity, Neuralink explores a new optimization space at the intersection of sparsity-driven algorithm and storage-level system co-design for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10407v1">PrediQL: Automated Testing of GraphQL APIs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 8 pages, two columns
    </div>
    <details class="paper-abstract">
      GraphQL's flexible query model and nested data dependencies expose APIs to complex, context-dependent vulnerabilities that are difficult to uncover using conventional testing tools. Existing fuzzers either rely on random payload generation or rigid mutation heuristics, failing to adapt to the dynamic structures of GraphQL schemas and responses. We present PrediQL, the first retrieval-augmented, LLM-guided fuzzer for GraphQL APIs. PrediQL combines large language model reasoning with adaptive feedback loops to generate semantically valid and diverse queries. It models the choice of fuzzing strategy as a multi-armed bandit problem, balancing exploration of new query structures with exploitation of past successes. To enhance efficiency, PrediQL retrieves and reuses execution traces, schema fragments, and prior errors, enabling self-correction and progressive learning across test iterations. Beyond input generation, PrediQL integrates a context-aware vulnerability detector that uses LLM reasoning to analyze responses, interpreting data values, error messages, and status codes to identify issues such as injection flaws, access-control bypasses, and information disclosure. Our evaluation across open-source and benchmark GraphQL APIs shows that PrediQL achieves significantly higher coverage and vulnerability discovery rates compared to state-of-the-art baselines. These results demonstrate that combining retrieval-augmented reasoning with adaptive fuzzing can transform API security testing from reactive enumeration to intelligent exploration.
    </details>
</div>
