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
- Part 8
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11813v1">Task-Aware Reduction for Scalable LLM-Database Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Preprint. Accepted for presentation at the Workshop on Language Models and Databases (LMD), co-located with CASCON 2025 (IEEE). The final version will appear in IEEE Xplore
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to data-intensive workflows, from database querying to developer observability. Yet the effectiveness of these systems is constrained by the volume, verbosity, and noise of real-world text-rich data such as logs, telemetry, and monitoring streams. Feeding such data directly into LLMs is costly, environmentally unsustainable, and often misaligned with task objectives. Parallel efforts in LLM efficiency have focused on model- or architecture-level optimizations, but the challenge of reducing upstream input verbosity remains underexplored. In this paper, we argue for treating the token budget of an LLM as an attention budget and elevating task-aware text reduction as a first-class design principle for language -- data systems. We position input-side reduction not as compression, but as attention allocation: prioritizing information most relevant to downstream tasks. We outline open research challenges for building benchmarks, designing adaptive reduction pipelines, and integrating token-budget--aware preprocessing into database and retrieval systems. Our vision is to channel scarce attention resources toward meaningful signals in noisy, data-intensive workflows, enabling scalable, accurate, and sustainable LLM--data integration.
    </details>
</div>
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10698v2">CrunchLLM: Multitask LLMs for Structured Business Reasoning and Outcome Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Predicting the success of start-up companies, defined as achieving an exit through acquisition or IPO, is a critical problem in entrepreneurship and innovation research. Datasets such as Crunchbase provide both structured information (e.g., funding rounds, industries, investor networks) and unstructured text (e.g., company descriptions), but effectively leveraging this heterogeneous data for prediction remains challenging. Traditional machine learning approaches often rely only on structured features and achieve moderate accuracy, while large language models (LLMs) offer rich reasoning abilities but struggle to adapt directly to domain-specific business data. We present \textbf{CrunchLLM}, a domain-adapted LLM framework for startup success prediction. CrunchLLM integrates structured company attributes with unstructured textual narratives and applies parameter-efficient fine-tuning strategies alongside prompt optimization to specialize foundation models for entrepreneurship data. Our approach achieves accuracy exceeding 80\% on Crunchbase startup success prediction, significantly outperforming traditional classifiers and baseline LLMs. Beyond predictive performance, CrunchLLM provides interpretable reasoning traces that justify its predictions, enhancing transparency and trustworthiness for financial and policy decision makers. This work demonstrates how adapting LLMs with domain-aware fine-tuning and structured--unstructured data fusion can advance predictive modeling of entrepreneurial outcomes. CrunchLLM contributes a methodological framework and a practical tool for data-driven decision making in venture capital and innovation policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04663v3">Debate, Deliberate, Decide (D3): A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) remains challenging due to inconsistency, bias, and the absence of transparent decision criteria in automated judging. We present Debate, Deliberate, Decide (D3), a cost-aware, adversarial multi-agent framework that orchestrates structured debate among role-specialized agents (advocates, a judge, and an optional jury) to produce reliable and interpretable evaluations. D3 instantiates two complementary protocols: (1) Multi-Advocate One-Round Evaluation (MORE), which elicits k parallel defenses per answer to amplify signal via diverse advocacy, and (2) Single-Advocate Multi-Round Evaluation (SAMRE) with budgeted stopping, which iteratively refines arguments under an explicit token budget and convergence checks. We develop a probabilistic model of score gaps that (i) characterizes reliability and convergence under iterative debate and (ii) explains the separation gains from parallel advocacy. Under mild assumptions, the posterior distribution of the round-r gap concentrates around the true difference and the probability of mis-ranking vanishes; moreover, aggregating across k advocates provably increases expected score separation. We complement theory with a rigorous experimental suite across MT-Bench, AlignBench, and AUTO-J, showing state-of-the-art agreement with human judgments (accuracy and Cohen's kappa), reduced positional and verbosity biases via anonymization and role diversification, and a favorable cost-accuracy frontier enabled by budgeted stopping. Ablations and qualitative analyses isolate the contributions of debate, aggregation, and anonymity. Together, these results establish D3 as a principled, practical recipe for reliable, interpretable, and cost-aware LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10331v1">LLM-Friendly Knowledge Representation for Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      We propose a practical approach by integrating Large Language Models (LLMs) with a framework designed to navigate the complexities of Airbnb customer support operations. In this paper, our methodology employs a novel reformatting technique, the Intent, Context, and Action (ICA) format, which transforms policies and workflows into a structure more comprehensible to LLMs. Additionally, we develop a synthetic data generation strategy to create training data with minimal human intervention, enabling cost-effective fine-tuning of our model. Our internal experiments (not applied to Airbnb products) demonstrate that our approach of restructuring workflows and fine-tuning LLMs with synthetic data significantly enhances their performance, setting a new benchmark for their application in customer support. Our solution is not only cost-effective but also improves customer support, as evidenced by both accuracy and manual processing time evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10329v1">End-to-end Automatic Speech Recognition and Speech Translation: Integration of Speech Foundational Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Speech Translation (ST) is a machine translation task that involves converting speech signals from one language to the corresponding text in another language; this task has two different approaches, namely the traditional cascade and the more recent end-to-end. This paper explores a combined end-to-end architecture of pre-trained speech encoders and Large Language Models (LLMs) for performing both Automatic Speech Recognition (ASR) and ST simultaneously. Experiments with the English-to-German language pair show that our best model not only can achieve better translation results than SeamlessM4T, a large foundational end-to-end, multi-modal translation model, but can also match the performance of a cascaded system with Whisper and NLLB, with up to a score gain of 8% in $\text{COMET}^{\text{DA}}_{22}$ metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09710v3">DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities to handle complex tasks. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions-varying in both source and difficulty. This heterogeneity introduces a key challenge: how to adaptively schedule training across distributions to optimize learning efficiency. In this paper, we present a principled curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose a distribution-level curriculum learning framework for RL-based LLM post-training, which leverages the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distrubutions. This approach prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration), yielding an adaptive and theoretically grounded training schedule. We instantiate our curriculum learning framework with GRPO as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distribution-aware curriculum strategies in LLM post-training. Code: https://github.com/ZhentingWang/DUMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10328v1">Are LLMs Empathetic to All? Investigating the Influence of Multi-Demographic Personas on a Model's Empathy</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 9 pages, 4 figures, 4 tables, EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models' (LLMs) ability to converse naturally is empowered by their ability to empathetically understand and respond to their users. However, emotional experiences are shaped by demographic and cultural contexts. This raises an important question: Can LLMs demonstrate equitable empathy across diverse user groups? We propose a framework to investigate how LLMs' cognitive and affective empathy vary across user personas defined by intersecting demographic attributes. Our study introduces a novel intersectional analysis spanning 315 unique personas, constructed from combinations of age, culture, and gender, across four LLMs. Results show that attributes profoundly shape a model's empathetic responses. Interestingly, we see that adding multiple attributes at once can attenuate and reverse expected empathy patterns. We show that they broadly reflect real-world empathetic trends, with notable misalignments for certain groups, such as those from Confucian culture. We complement our quantitative findings with qualitative insights to uncover model behaviour patterns across different demographic groups. Our findings highlight the importance of designing empathy-aware LLMs that account for demographic diversity to promote more inclusive and equitable model behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10281v1">ArtPerception: ASCII Art-based Jailbreak on LLMs with Recognition Pre-test</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 30 pages, 22 figures. This preprint has been accepted for publication in Elsevier JOURNAL OF NETWORK AND COMPUTER APPLICATIONS (JNCA)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into computer applications has introduced transformative capabilities but also significant security challenges. Existing safety alignments, which primarily focus on semantic interpretation, leave LLMs vulnerable to attacks that use non-standard data representations. This paper introduces ArtPerception, a novel black-box jailbreak framework that strategically leverages ASCII art to bypass the security measures of state-of-the-art (SOTA) LLMs. Unlike prior methods that rely on iterative, brute-force attacks, ArtPerception introduces a systematic, two-phase methodology. Phase 1 conducts a one-time, model-specific pre-test to empirically determine the optimal parameters for ASCII art recognition. Phase 2 leverages these insights to launch a highly efficient, one-shot malicious jailbreak attack. We propose a Modified Levenshtein Distance (MLD) metric for a more nuanced evaluation of an LLM's recognition capability. Through comprehensive experiments on four SOTA open-source LLMs, we demonstrate superior jailbreak performance. We further validate our framework's real-world relevance by showing its successful transferability to leading commercial models, including GPT-4o, Claude Sonnet 3.7, and DeepSeek-V3, and by conducting a rigorous effectiveness analysis against potential defenses such as LLaMA Guard and Azure's content filters. Our findings underscore that true LLM security requires defending against a multi-modal space of interpretations, even within text-only inputs, and highlight the effectiveness of strategic, reconnaissance-based attacks. Content Warning: This paper includes potentially harmful and offensive model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10276v1">Lost in the Middle: An Emergent Property from Information Retrieval Demands in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) often degrades when crucial information is in the middle of a long context, a "lost-in-the-middle" phenomenon that mirrors the primacy and recency effects in human memory. We propose that this behavior is not simply a flaw indicative of information loss but an adaptation to different information retrieval demands during pre-training: some tasks require uniform recall across the entire input (a long-term memory demand), while others prioritize the most recent information (a short-term memory demand). Consistent with this view, we show that this U-shaped performance curve emerges when LLMs (GPT-2 and Llama variants) are trained from scratch on two simple human memory paradigms simulating long-term and short-term memory demands. Our analysis reveals that while the recency effect directly aligns with short-term memory demand in the training data, the primacy effect is induced by the uniform long-term memory demand and is additionally influenced by the model's autoregressive properties and the formation of attention sinks. Our main findings from simple human memory paradigms also generalize to a sequence completion task, which more closely resembles the next-token prediction process in LLM pre-training. Together, our findings reveal how information retrieval demands, model architecture, and structural attention dynamics during model training can jointly produce positional bias observed in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10271v1">MetaBreak: Jailbreaking Online LLM Services via Special Token Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Unlike regular tokens derived from existing text corpora, special tokens are artificially created to annotate structured conversations during the fine-tuning process of Large Language Models (LLMs). Serving as metadata of training data, these tokens play a crucial role in instructing LLMs to generate coherent and context-aware responses. We demonstrate that special tokens can be exploited to construct four attack primitives, with which malicious users can reliably bypass the internal safety alignment of online LLM services and circumvent state-of-the-art (SOTA) external content moderation systems simultaneously. Moreover, we found that addressing this threat is challenging, as aggressive defense mechanisms-such as input sanitization by removing special tokens entirely, as suggested in academia-are less effective than anticipated. This is because such defense can be evaded when the special tokens are replaced by regular ones with high semantic similarity within the tokenizer's embedding space. We systemically evaluated our method, named MetaBreak, on both lab environment and commercial LLM platforms. Our approach achieves jailbreak rates comparable to SOTA prompt-engineering-based solutions when no content moderation is deployed. However, when there is content moderation, MetaBreak outperforms SOTA solutions PAP and GPTFuzzer by 11.6% and 34.8%, respectively. Finally, since MetaBreak employs a fundamentally different strategy from prompt engineering, the two approaches can work synergistically. Notably, empowering MetaBreak on PAP and GPTFuzzer boosts jailbreak rates by 24.3% and 20.2%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23841v2">SkewRoute: Training-Free LLM Routing for Knowledge Graph Retrieval-Augmented Generation via Score Skewness of Retrieved Context</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Large language models excel at many tasks but often incur high inference costs during deployment. To mitigate hallucination, many systems use a knowledge graph to enhance retrieval-augmented generation (KG-RAG). However, the large amount of retrieved knowledge contexts increase these inference costs further. A promising solution to balance performance and cost is LLM routing, which directs simple queries to smaller LLMs and complex ones to larger LLMs. However, no dedicated routing methods currently exist for RAG, and existing training-based routers face challenges scaling to this domain due to the need for extensive training data. We observe that the score distributions produced by the retrieval scorer strongly correlate with query difficulty. Based on this, we propose an extremely simple yet effective routing framework, the first specifically designed for KG-RAG that efficiently balances performance and cost in a plug-and-play manner. It delivers over 3x higher routing effectiveness while reducing runtime to less than 0.001x compared to existing methods. Our code is available at https://github.com/hrwang00/SkewRoute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10241v1">ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10238v1">The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become foundational tools in natural language processing, powering a wide range of applications and research. Many studies have shown that LLMs share significant similarities with the human brain. Recent neuroscience research has found that a small subset of biological neurons in the human brain are crucial for core cognitive functions, which raises a fundamental question: do LLMs also contain a small subset of critical neurons? In this paper, we investigate this question by proposing a Perturbation-based Causal Identification of Critical Neurons method to systematically locate such critical neurons in LLMs. Our findings reveal three key insights: (1) LLMs contain ultra-sparse critical neuron sets. Disrupting these critical neurons can cause a 72B-parameter model with over 1.1 billion neurons to completely collapse, with perplexity increasing by up to 20 orders of magnitude; (2) These critical neurons are not uniformly distributed, but tend to concentrate in the outer layers, particularly within the MLP down\_proj components; (3) Performance degradation exhibits sharp phase transitions, rather than a gradual decline, when these critical neurons are disrupted. Through comprehensive experiments across diverse model architectures and scales, we provide deeper analysis of these phenomena and their implications for LLM robustness and interpretability. These findings can offer guidance for developing more robust model architectures and improving deployment security in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02393v2">AP2O: Correcting LLM-Generated Code Errors Type by Type Like Humans via Adaptive Progressive Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      LLMs' code generation capabilities have yielded substantial improvements in the effectiveness of programming tasks. However, LLM-generated code still suffers from compilation and runtime errors. Existing offline preference optimization methods primarily focus on enhancing LLMs' coding abilities using pass/fail signals in the preference data, overlooking the deep-level error types in the failed codes. To address this, we propose Adaptively Progressive Preference Optimization (AP2O) for coding (i.e., AP2O-Coder), a method that guides LLMs adaptively and methodically to reduce code errors for code generation. Specifically, we construct an error notebook from failed codes and progressively optimize the LLM to correct errors type by type. Furthermore, we adaptively replay error types to tailor to the LLM's changing weaknesses throughout the training process. Through extensive experiments on both code and general LLMs (Llama, Qwen, and DeepSeek series) with parameters ranging from 0.5B to 34B, our AP2O-Coder improves code generation performance by up to 3% in pass@k while using less preference data. Code: https://github.com/TsingZ0/AP2O
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04214v2">Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment frm Heterogeneous Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      We study deploying large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs), where aligning traveler affordability and hotel profitability directly affects bookings, partner relationships, and access to travel. The agent must follow a Standard Operating Procedure (SOP) while conducting multi-turn persuasion, interpreting colloquial inputs, and adhering to guardrails (no over-promising, no hallucinations). Conventional post-training -- supervised fine-tuning (SFT) or single-source reward optimization -- overfits scripts, misses nuanced persuasive style, and fails to enforce verifiable business constraints. We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training framework that aligns an LLM with heterogeneous rewards: a preference-trained reward model (RM) for dense human alignment, a reward judge (RJ) for high-level persuasive behavior and SOP compliance, and programmatic reward functions (RF) for deterministic checks on numerics, formatting, and guardrails. A straightforward enhancement mechanism is proposed to combine the RM with RJ and RF signals to curb reward hacking and improve negotiation quality. In production-style evaluations -- approximately 150 turns from real dialogues and 225 turns from curated bad-case dialogues -- REPO lifts average dialogue rating to 4.63: +1.20 over base, +0.83 over Direct Preference Optimization (DPO); +0.33 over Group Relative Policy Optimization (GRPO), increases the share of conversations with at least one excellent response to 66.67% (+23.34 percentage points over GRPO), and achieves a 93.33% bad-case fix rate with 75.56% clean fixes, outperforming SFT, DPO, PPO, and GRPO. We also observe emergent capabilities -- proactive empathy, localized reasoning, calibrated tactics -- that surpass gold annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10225v1">ISAAC: Intelligent, Scalable, Agile, and Accelerated CPU Verification via LLM-aided FPGA Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Functional verification is a critical bottleneck in integrated circuit development, with CPU verification being especially time-intensive and labour-consuming. Industrial practice relies on differential testing for CPU verification, yet faces bottlenecks at nearly each stage of the framework pipeline: front-end stimulus generation lacks micro-architectural awareness, yielding low-quality and redundant tests that impede coverage closure and miss corner cases. Meanwhile, back-end simulation infrastructure, even with FPGA acceleration, often stalls on long-running tests and offers limited visibility, delaying feedback and prolonging the debugging cycle. Here, we present ISAAC, a full-stack, Large Language Model (LLM)-aided CPU verification framework with FPGA parallelism, from bug categorisation and stimulus generation to simulation infrastructure. To do so, we presented a multi-agent stimulus engine in ISAAC's front-end, infused with micro-architectural knowledge and historical bug patterns, generating highly targeted tests that rapidly achieve coverage goals and capture elusive corner cases. In ISAAC's back-end, we introduce a lightweight forward-snapshot mechanism and a decoupled co-simulation architecture between the Instruction Set Simulator (ISS) and the Design Under Test (DUT), enabling a single ISS to drive multiple DUTs in parallel. By eliminating long-tail test bottlenecks and exploiting FPGA parallelism, the simulation throughput is significantly improved. As a demonstration, we used ISAAC to verify a mature CPU that has undergone multiple successful tape-outs. Results show up to 17,536x speed-up over software RTL simulation, while detecting several previously unknown bugs, two of which are reported in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10223v1">You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in specialized domains such as finance, medicine, and agriculture, where they face significant distribution shifts from their training data. Domain-specific fine-tuning can mitigate this challenge but relies on high-quality labeled data that is expensive and slow to collect in expertise-limited settings. We study label-free test-time adaptation for language models and present SyTTA, an inference-time framework that adapts models on-the-fly without additional supervision. SyTTA couples two complementary uncertainty signals that arise under distribution shift: input-side perplexity, indicating mismatch with domain-specific terminology and patterns, and output-side predictive entropy, indicating diffuse and unstable token probabilities during generation. Across diverse model architectures and domain-specific benchmarks, SyTTA delivers consistent gains. Notably, on agricultural question answering, SyTTA improves Rouge-LSum by over 120% on Qwen-2.5-7B with only 4 extra tokens per query. These results show that effective test-time adaptation for language models is achievable without labeled examples, supporting deployment in label-scarce domains. The code will be made available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14226v3">"Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Recently released LLMs have strong multilingual \& multimodal capabilities. Model vulnerabilities are exposed using audits and red-teaming efforts. Existing efforts have focused primarily on the English language; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially for multimodal contexts. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also present an extension to a current jailbreak-template-based strategy and propose a novel template, showing higher effectiveness than baselines. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. We achieve a 99\% Attack Success Rate for text generation and 78\% for image generation, with Attack Relevance Rate of 100\% for text generation and 96\% for image generation for the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words. \textit{\textbf{Warning: This paper contains examples of potentially harmful and offensive content.}}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10201v1">RLFR: Extending Reinforcement Learning for LLMs with Flow Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Project Website: https://jinghaoleven.github.io/RLFR/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a promising framework for improving reasoning abilities in Large Language Models (LLMs). However, policy optimized with binary verification prone to overlook potential valuable exploration in reasoning trajectory. In view of heavy annotation cost of golden Process Reward Models (PRMs), recent works attempt using auxiliary signals for reward shaping of process tokens, involving entropy and likelihood collected from logit space. In this work, we offer a novel perspective on shaping RLVR with flow rewards derived from latent space, and propose RLFR, where the flow fields of model latents are constructed from either off-policy high-quality data and on-policy rejection sampling data, and the velocity deviations of policy latents within it are quantified to serve as a reward signal. RLFR first demonstrates that a well-established flow field can be a sound environment for reward signal collection, highlighting the expressive latent space is much underexplored. Moreover, RLFR is able to compress any off-policy expert data as reference for constituting reward signals, and we show that the efficient context dependence compressed within the hidden states are utilized, rather than individual token-level denotation for context comprehending. Experiments on both language and multimodal reasoning benchmarks demonstrate the reliability of flow rewards, and suggesting a promising paradigm for reward shaping with auxiliary signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20371v3">DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Accepted by EMNLP2025-main
    </div>
    <details class="paper-abstract">
      Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory, the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT, remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompt strategies, and (3) we design precise disambiguation metrics, and study the efficacy of various prompt strategies on multiple state-of-the-art LLMs. We conduct comprehensive experiments across 4 language pairs and 13 domains, our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10390v2">Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Jailbreaking commercial black-box models is one of the most challenging and serious security threats today. Existing attacks achieve certain success on non-reasoning models but perform limitedly on the latest reasoning models. We discover that carefully crafted developer messages can markedly boost jailbreak effectiveness. Building on this, we propose two developer-role-based attacks: D-Attack, which enhances contextual simulation, and DH-CoT, which strengthens attacks with deceptive chain-of-thought. In experiments, we further diccover that current red-teaming datasets often contain samples unsuited for measuring attack gains: prompts that fail to trigger defenses, prompts where malicious content is not the sole valid output, and benign prompts. Such data hinders accurate measurement of the true improvement brought by an attack method. To address this, we introduce MDH, a Malicious content Detection approach combining LLM-based screening with Human verification to balance accuracy and cost, with which we clean data and build the RTA dataset series. Experiments demonstrate that MDH reliably filters low-quality samples and that developer messages significantly improve jailbreak attack success. Codes, datasets, and other results will be released in https://github.com/AlienZhang1996/DH-CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10179v1">LLMs are All You Need? Improving Fuzz Testing for MOJO with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has revolutionized software testing, particularly fuzz testing, by automating the generation of diverse and effective test inputs. This advancement holds great promise for improving software reliability. Meanwhile, the introduction of MOJO, a high-performance AI programming language blending Python's usability with the efficiency of C and C++, presents new opportunities to enhance AI model scalability and programmability. However, as a new language, MOJO lacks comprehensive testing frameworks and a sufficient corpus for LLM-based testing, which exacerbates model hallucination. In this case, LLMs will generate syntactically valid but semantically incorrect code, significantly reducing the effectiveness of fuzz testing. To address this challenge, we propose MOJOFuzzer, the first adaptive LLM-based fuzzing framework designed for zero-shot learning environments of emerging programming languages. MOJOFuzzer integrates a mutil-phase framework that systematically eliminates low-quality generated inputs before execution, significantly improving test case validity. Furthermore, MOJOFuzzer dynamically adapts LLM prompts based on runtime feedback for test case mutation, enabling an iterative learning process that continuously enhances fuzzing efficiency and bug detection performance. Our experimental results demonstrate that MOJOFuzzer significantly enhances test validity, API coverage, and bug detection performance, outperforming traditional fuzz testing and state-of-the-art LLM-based fuzzing approaches. Using MOJOFuzzer, we have conducted a first large-scale fuzz testing evaluation of MOJO, uncorvering 13 previous unknown bugs. This study not only advances the field of LLM-driven software testing but also establishes a foundational methodology for leveraging LLMs in the testing of emerging programming languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21240v2">Tree Search for LLM Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14119v3">CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 NeurIPS 2025; 10 pages of main text; 25 pages of appendices. Website - https://cuhk-arise.github.io/CodeCrash/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong capabilities in code-related tasks, but their robustness in code reasoning under perturbations remains underexplored. We introduce CodeCrash, a stress-testing framework with 1,279 questions from CruxEval and LiveCodeBench, designed to evaluate reasoning reliability under structural perturbations and misleading natural language (NL) contexts. Through a systematic evaluation of 17 LLMs, we find that models often shortcut reasoning by over-relying on NL cues, leading to an average performance degradation of 23.2% in output prediction tasks. Even with Chain-of-Thought reasoning, models on average still have a 13.8% drop due to distractibility and rationalization, revealing a lack of critical reasoning capability to distinguish the actual code behaviors. While Large Reasoning Models with internal reasoning mechanisms improve robustness by fostering critical thinking, plausible yet incorrect hints can trigger pathological self-reflection, causing 2-3 times token consumption and even catastrophic cognitive dissonance in extreme cases for QwQ-32B. We refer to this phenomenon as Reasoning Collapse. CodeCrash provides a rigorous benchmark for evaluating robustness in code reasoning, guiding future research and development toward more reliable and resilient models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10138v1">Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Information extraction from copy-heavy documents, characterized by massive volumes of structurally similar content, represents a critical yet understudied challenge in enterprise document processing. We present a systematic framework that strategically combines OCR engines with Large Language Models (LLMs) to optimize the accuracy-efficiency trade-off inherent in repetitive document extraction tasks. Unlike existing approaches that pursue universal solutions, our method exploits document-specific characteristics through intelligent strategy selection. We implement and evaluate 25 configurations across three extraction paradigms (direct, replacement, and table-based) on identity documents spanning four formats (PNG, DOCX, XLSX, PDF). Through table-based extraction methods, our adaptive framework delivers outstanding results: F1=1.0 accuracy with 0.97s latency for structured documents, and F1=0.997 accuracy with 0.6 s for challenging image inputs when integrated with PaddleOCR, all while maintaining sub-second processing speeds. The 54 times performance improvement compared with multimodal methods over naive approaches, coupled with format-aware routing, enables processing of heterogeneous document streams at production scale. Beyond the specific application to identity extraction, this work establishes a general principle: the repetitive nature of copy-heavy tasks can be transformed from a computational burden into an optimization opportunity through structure-aware method selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10131v1">Proof Strategy Extraction from LLMs for Enhancing Symbolic Provers</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      One important approach to software verification is interactive theorem proving. However, writing formal proofs often requires substantial human effort, making proof automation highly important. Traditionally, proof automation has relied on symbolic provers. Recently, large language models (LLMs) have demonstrated strong capabilities in theorem proving, complementing symbolic provers. Nonetheless, prompting LLMs can be expensive and may pose security risks for confidential codebases. As a result, purely symbolic approaches remain important even in the LLM era, as they are cost-effective, secure, and complement the strengths of LLMs. Motivated by these considerations, we ask a new research question: can we extract the internal strategies of LLMs to enhance the capabilities of symbolic provers? As an initial attempt to answer this question, we propose Strat2Rocq, which extracts proof strategies from LLMs and formalizes them as lemmas in Rocq. These lemmas are accessible to symbolic provers such as CoqHammer. With the addition of these LLM-extracted lemmas, CoqHammer is able to prove more theorems. The knowledge extraction process involves analyzing the proof trajectories of LLMs on a training set of proved theorems. For each theorem, we prompt the LLM to generate a natural language proof, then ask it to summarize this proof into formalized lemmas with proofs. We also employ a standard agentic approach to mitigate errors during formalization. Our evaluation demonstrates that, on open-source Rocq projects for software verification, Strat2Rocq enhances the success rate of CoqHammer by 13.41%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10119v1">IntrinTrans: LLM-based Intrinsic Code Translator for RISC-V Vector</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The use of intrinsic functions to exploit hardware-specific capabilities is an important approach for optimizing library performance. Many mainstream libraries implement a large number of vectorized algorithms on Arm or x86 SIMD intrinsic functions. With the rapid expansion of the RISC-V hardware-software ecosystem, there is a growing demand for support of the RISC-V Vector (RVV) extension. Translating existing vectorized intrinsic code onto RVV intrinsics is a practical and effective approach. However, current cross-architecture translation largely relies on manual rewriting, which is time-consuming and error-prone. Furthermore, while some rule-based methods can reduce the need for manual intervention, their translation success rate is limited by incomplete rule coverage and syntactic constraints, and the performance suffers from inadequate utilization of RVV-specific features. We present IntrinTrans, a LLM-based multi-agent approach that utilizes compile-and-test feedback to translate intrinsic code across architectures automatically, and further optimizes the generated RVV intrinsics using register-usage information derived from liveness analysis. To evaluate the effectiveness of our approach, we collected 34 vectorized algorithm cases from open-source libraries. Each case includes an Arm Neon intrinsics implementation and a RVV intrinsics implementation contributed by the open-source community, together with correctness and performance tests. Our experiments show that advanced LLMs produce semantically correct RISC-V Vector intrinsics in most cases within a limited number of iterations, and in some cases achieve up to 5.93x the performance of the native implementation from the open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10066v1">OBsmith: Testing JavaScript Obfuscator using LLM-powered sketching</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      JavaScript obfuscators are widely deployed to protect intellectual property and resist reverse engineering, yet their correctness has been largely overlooked compared to performance and resilience. Existing evaluations typically measure resistance to deobfuscation, leaving the critical question of whether obfuscators preserve program semantics unanswered. Incorrect transformations can silently alter functionality, compromise reliability, and erode security-undermining the very purpose of obfuscation. To address this gap, we present OBsmith, a novel framework to systematically test JavaScript obfuscators using large language models (LLMs). OBsmith leverages LLMs to generate program sketches abstract templates capturing diverse language constructs, idioms, and corner cases-which are instantiated into executable programs and subjected to obfuscation under different configurations. Besides LLM-powered sketching, OBsmith also employs a second source: automatic extraction of sketches from real programs. This extraction path enables more focused testing of project specific features and lets developers inject domain knowledge into the resulting test cases. OBsmith uncovers 11 previously unknown correctness bugs. Under an equal program budget, five general purpose state-of-the-art JavaScript fuzzers (FuzzJIT, Jsfunfuzz, Superion, DIE, Fuzzilli) failed to detect these issues, highlighting OBsmith's complementary focus on obfuscation induced misbehavior. An ablation shows that all components except our generic MRs contribute to at least one bug class; the negative MR result suggests the need for obfuscator-specific metamorphic relations. Our results also seed discussion on how to balance obfuscation presets and performance cost. We envision OBsmith as an important step towards automated testing and quality assurance of obfuscators and other semantic-preserving toolchains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10028v1">Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The rapid advancement of Low-Altitude Economy Networks (LAENets) has enabled a variety of applications, including aerial surveillance, environmental sensing, and semantic data collection. To support these scenarios, unmanned aerial vehicles (UAVs) equipped with onboard vision-language models (VLMs) offer a promising solution for real-time multimodal inference. However, ensuring both inference accuracy and communication efficiency remains a significant challenge due to limited onboard resources and dynamic network conditions. In this paper, we first propose a UAV-enabled LAENet system model that jointly captures UAV mobility, user-UAV communication, and the onboard visual question answering (VQA) pipeline. Based on this model, we formulate a mixed-integer non-convex optimization problem to minimize task latency and power consumption under user-specific accuracy constraints. To solve the problem, we design a hierarchical optimization framework composed of two parts: (i) an Alternating Resolution and Power Optimization (ARPO) algorithm for resource allocation under accuracy constraints, and (ii) a Large Language Model-augmented Reinforcement Learning Approach (LLaRA) for adaptive UAV trajectory optimization. The large language model (LLM) serves as an expert in refining reward design of reinforcement learning in an offline fashion, introducing no additional latency in real-time decision-making. Numerical results demonstrate the efficacy of our proposed framework in improving inference performance and communication efficiency under dynamic LAENet conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16331v3">Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10010v1">SLEAN: Simple Lightweight Ensemble Analysis Network for Multi-Provider LLM Coordination: Design, Implementation, and Vibe Coding Bug Investigation Case Study</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 14 pages, 4 figures, 6 tables, link to code repo
    </div>
    <details class="paper-abstract">
      We present SLEAN (Simple Lightweight Ensemble Analysis Network), a deterministic framework for coordinating multiple LLM providers through text-based prompt orchestration. Unlike complex multi-agent systems requiring specialized infrastructure, SLEAN operates as a simple prompt bridge between LLMs using .txt templates, requiring no deep technical knowledge for deployment. The three-phase protocol formed by independent analysis, cross-critique, and arbitration, filters harmful AI-generated code suggestions before production deployment, addressing how AI-assisted debugging increasingly produces modifications that introduce unnecessary complexity, break existing functionality, or address problems. Evaluating 15 software bugs, we analyzed 69 AI-generated fix propositions. SLEAN's filtering accepted 22 fixes (31.9%, 95% CI 20.9-42.9%) while rejecting 47 that would have been harmful if applied verbatim. The arbitration process reduced code change surface by 83-90% relative to raw AI outputs, enforcing minimal causal edits over scope-expanding modifications. Minimal Type 2 inputs proved more efficient than detailed Type 1 inputs, requiring 2.85 versus 3.56 propositions per accepted fix (35.1% versus 28.1% acceptance, about a 20% efficiency gain). Agreement between AI systems showed weak correlation with fix quality: high convergence (at least 80%) occurred in 4 of 15 cases and improved acceptance by only 2.4% points; arbitration appeared only at exactly 10% convergence in 2 of 15 cases, although low convergence alone did not necessitate arbitration. The file-driven, provider-agnostic architecture enables deployment without specialized coding expertise, making it applicable to security auditing, code review, document verification, and other domains requiring reliable multi-provider synthesis with end-to-end auditability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10009v1">Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Reasoning-augmented search agents, such as Search-R1, are trained to reason, search, and generate the final answer iteratively. Nevertheless, due to their limited capabilities in reasoning and search, their performance on multi-hop QA benchmarks remains far from satisfactory. To handle complex or compound queries, we train an LLM-based search agent with the native capability of query expansion through reinforcement learning. In each turn, our search agent proposes several query variants, which are searched simultaneously to cover more relevant information. Meanwhile, given limited post-training data and computing resources, it is very challenging for a search agent to master multiple tasks, including query generation, retrieved information understanding, and answer generation. Therefore, we propose incorporating a pre-trained squeezer model that helps the search agent understand the retrieved documents, allowing the search agent to focus on query generation for high retrieval recall. With the assistance of the squeezer model, we discover that even a small-scale 3B LLM can demonstrate a strong capability of query expansion and achieve state-of-the-art accuracy on the multi-hop QA benchmarks. To be specific, our experiments across seven question-answering benchmarks demonstrate that our method, named ExpandSearch, achieves an average improvement of 4.4% compared to state-of-the-art baselines, with strong gains on multi-hop reasoning tasks requiring diverse evidence aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17262v3">Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 24 pages,6 figures
    </div>
    <details class="paper-abstract">
      The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for efficient resource allocation. This is challenged by: 1) the emergence phenomenon, where metrics become meaningful only after extensive training, hindering prediction by smaller models; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby establishing a more stable and predictable support subset through the exclusion of tasks exhibiting non-emergent behavior or irregular scaling. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.36% average prediction error across eight key LLM benchmarks, offering actionable insights for resource allocation and training monitoring of LLMs pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10002v1">Deliberative Dynamics and Value Alignment in LLM Debates</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in sensitive everyday contexts - offering personal advice, mental health support, and moral guidance - understanding their elicited values in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings where values emerge through dialogue, revision, and consensus. We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's "Am I the Asshole" community. We use both synchronous (parallel responses) and round-robin (sequential responses) formats to test order effects and verdict revision. Our findings show striking behavioral differences. In the synchronous setting, GPT showed strong inertia (0.6-3.1% revision rates) while Claude and Gemini were far more flexible (28-41%). Value patterns also diverged: GPT emphasized personal autonomy and direct communication, while Claude and Gemini prioritized empathetic dialogue. Certain values proved especially effective at driving verdict changes. We further find that deliberation format had a strong impact on model behavior: GPT and Gemini stood out as highly conforming relative to Claude, with their verdict behavior strongly shaped by order effects. These results show how deliberation format and model-specific behaviors shape moral reasoning in multi-turn interactions, underscoring that sociotechnical alignment depends on how systems structure dialogue as much as on their outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v2">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09988v1">Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Deliberative tree search is a cornerstone of modern Large Language Model (LLM) research, driving the pivot from brute-force scaling toward algorithmic efficiency. This single paradigm unifies two critical frontiers: \textbf{Test-Time Scaling (TTS)}, which deploys on-demand computation to solve hard problems, and \textbf{Self-Improvement}, which uses search-generated data to durably enhance model parameters. However, this burgeoning field is fragmented and lacks a common formalism, particularly concerning the ambiguous role of the reward signal -- is it a transient heuristic or a durable learning target? This paper resolves this ambiguity by introducing a unified framework that deconstructs search algorithms into three core components: the \emph{Search Mechanism}, \emph{Reward Formulation}, and \emph{Transition Function}. We establish a formal distinction between transient \textbf{Search Guidance} for TTS and durable \textbf{Parametric Reward Modeling} for Self-Improvement. Building on this formalism, we introduce a component-centric taxonomy, synthesize the state-of-the-art, and chart a research roadmap toward more systematic progress in creating autonomous, self-improving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11574v3">Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 23pages
    </div>
    <details class="paper-abstract">
      Low-bit post-training quantization (PTQ) is a practical route to deploy reasoning-capable LLMs under tight memory and latency budgets, yet it can markedly impair mathematical reasoning (drops up to 69.81% in our harder settings). We address two deployment-critical questions with process-level precision: Where along a step-structured solution does degradation first arise? How to mitigate it while staying in the low-bit regime? Across widely used PTQ methods (AWQ, GPTQ, SmoothQuant), open-source model families (Qwen, LLaMA; 0.5--7B), and math reasoning benchmarks (GSM8K, MATH, AIME), we perform format-aligned chain-of-thought with step-aligned attribution and uncover two robust regularities: (i) PTQ disproportionately elevates method and execution errors relative to high-level conceptual mistakes; and (ii) failures emerge early, with the first vulnerable step flipping and cascading to the final answer. These regularities suggest a general intervention principle: restore local token-level margins exactly at the earliest failure frontier. We instantiate this principle as a lightweight measure$\rightarrow$locate$\rightarrow$restore loop that operates directly on the quantized model: detect the first faulty step, construct our "Silver Bullet" datasets, and apply small-scale supervised/preference tuning. In our settings, as few as 332 curated examples and 3--5 minutes of compute on a single GPU recover 4-bit weight math reasoning toward the full-precision baseline while preserving PTQ efficiency. Our framework is quantizer- and architecture-agnostic within the evaluated regimes, and turns low-bit degradation from a global accuracy problem into a local, reproducible process intervention.
    </details>
</div>
