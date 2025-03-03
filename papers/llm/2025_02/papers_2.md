# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
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

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18449v1">SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11709v3">Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential in software development, especially for issue resolution. However, despite their widespread use, significant challenges persist in the quality of LLM responses to issue resolution queries. LLM interactions often yield incorrect, incomplete, or ambiguous information, largely due to knowledge gaps in prompt design, which can lead to unproductive exchanges and reduced developer productivity. In this paper, we analyze 433 developer-ChatGPT conversations within GitHub issue threads to examine the impact of prompt knowledge gaps and conversation styles on issue resolution. We identify four main knowledge gaps in developer prompts: Missing Context, Missing Specifications, Multiple Context, and Unclear Instructions. Assuming that conversations within closed issues contributed to successful resolutions while those in open issues did not, we find that ineffective conversations contain knowledge gaps in 44.6% of prompts, compared to only 12.6% in effective ones. Additionally, we observe seven distinct conversational styles, with Directive Prompting, Chain of Thought, and Responsive Feedback being the most prevalent. We find that knowledge gaps are present in all styles of conversations, with Missing Context being the most repeated challenge developers face in issue-resolution conversations. Based on our analysis, we identify key textual and code-related heuristics (Specificity, Contextual Richness, and Clarity) that are associated with successful issue closure and help assess prompt quality. These heuristics lay the foundation for an automated tool that can dynamically flag unclear prompts and suggest structured improvements. To test feasibility, we developed a lightweight browser extension prototype for detecting prompt gaps, that can be easily adapted to other tools within developer workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10563v2">Accelerating Unbiased LLM Evaluation via Synthetic Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      When developing new large language models (LLMs), a key step is evaluating their final performance, often by computing the win-rate against a reference model based on external feedback. Human feedback is the gold standard, particularly for capturing nuanced qualities like coherence, readability, and alignment with human expectations. However, human evaluations are costly -- even for large tech companies -- and when conducted with active users, they may negatively impact user experience. A promising alternative is synthetic feedback, where evaluations are conducted by other large language models, including reward models. While this eliminates the need for costly human annotations, it introduces biases that may distort the evaluation process. In this work, we propose a statistically principled framework that integrates human and synthetic feedback to reduce reliance on human annotations while maintaining unbiased win-rate calculations. Our experiments demonstrate a reduction in human annotations by up to 12.2% with an off-the-shelf synthetic evaluator and up to 24.8% with a finetuned variant. Apart from being generalizable, scalable, and free of hyper-parameter tuning, our method offers predictable annotation savings, which can be estimated based on data-dependent characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18414v1">GLEAN: Generalized Category Discovery with Diverse and Quality-Enhanced LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Generalized Category Discovery (GCD) is a practical and challenging open-world task that aims to recognize both known and novel categories in unlabeled data using limited labeled data from known categories. Due to the lack of supervision, previous GCD methods face significant challenges, such as difficulty in rectifying errors for confusing instances, and inability to effectively uncover and leverage the semantic meanings of discovered clusters. Therefore, additional annotations are usually required for real-world applicability. However, human annotation is extremely costly and inefficient. To address these issues, we propose GLEAN, a unified framework for generalized category discovery that actively learns from diverse and quality-enhanced LLM feedback. Our approach leverages three different types of LLM feedback to: (1) improve instance-level contrastive features, (2) generate category descriptions, and (3) align uncertain instances with LLM-selected category descriptions. Extensive experiments demonstrate the superior performance of \MethodName over state-of-the-art models across diverse datasets, metrics, and supervision settings. Our code is available at https://github.com/amazon-science/Glean.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18413v1">When Benchmarks Talk: Re-Evaluating Code LLMs with Interactive Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Programming is a fundamentally interactive process, yet coding assistants are often evaluated using static benchmarks that fail to measure how well models collaborate with users. We introduce an interactive evaluation pipeline to examine how LLMs incorporate different types of feedback in a collaborative setting. Specifically, we perturb static coding benchmarks so that the code model must interact with a simulated user to retrieve key information about the problem. We find that interaction significantly affects model performance, as the relative rankings of 10 models across 3 datasets often vary between static and interactive settings, despite models being fairly robust to feedback that contains errors. We also observe that even when different feedback types are equally effective with respect to performance, they can impact model behaviors such as (1) how models respond to higher- vs. lower-quality feedback and (2) whether models prioritize aesthetic vs. functional edits. Our work aims to "re-evaluate" model coding capabilities through an interactive lens toward bridging the gap between existing evaluations and real-world usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18389v1">Monte Carlo Temperature: a robust sampling strategy for LLM's uncertainty quantification methods</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) in Large Language Models (LLMs) is essential for their safe and reliable deployment, particularly in critical applications where incorrect outputs can have serious consequences. Current UQ methods typically rely on querying the model multiple times using non-zero temperature sampling to generate diverse outputs for uncertainty estimation. However, the impact of selecting a given temperature parameter is understudied, and our analysis reveals that temperature plays a fundamental role in the quality of uncertainty estimates. The conventional approach of identifying optimal temperature values requires expensive hyperparameter optimization (HPO) that must be repeated for each new model-dataset combination. We propose Monte Carlo Temperature (MCT), a robust sampling strategy that eliminates the need for temperature calibration. Our analysis reveals that: 1) MCT provides more robust uncertainty estimates across a wide range of temperatures, 2) MCT improves the performance of UQ methods by replacing fixed-temperature strategies that do not rely on HPO, and 3) MCT achieves statistical parity with oracle temperatures, which represent the ideal outcome of a well-tuned but computationally expensive HPO process. These findings demonstrate that effective UQ can be achieved without the computational burden of temperature parameter calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18387v1">How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 31 pages, 9 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Search plays a fundamental role in problem-solving across various domains, with most real-world decision-making problems being solvable through systematic search. Drawing inspiration from recent discussions on search and learning, we systematically explore the complementary relationship between search and Large Language Models (LLMs) from three perspectives. First, we analyze how learning can enhance search efficiency and propose Search via Learning (SeaL), a framework that leverages LLMs for effective and efficient search. Second, we further extend SeaL to SeaL-C to ensure rigorous completeness during search. Our evaluation across three real-world planning tasks demonstrates that SeaL achieves near-perfect accuracy while reducing search spaces by up to 99.1% compared to traditional approaches. Finally, we explore how far LLMs are from real search by investigating whether they can develop search capabilities independently. Our analysis reveals that while current LLMs struggle with efficient search in complex problems, incorporating systematic search strategies significantly enhances their problem-solving capabilities. These findings not only validate the effectiveness of our approach but also highlight the need for improving LLMs' search abilities for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17962v5">Crafting Customisable Characters with LLMs: Introducing SimsChat, a Persona-Driven Role-Playing Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable ability to comprehend instructions and generate human-like text, enabling sophisticated agent simulation beyond basic behavior replication. However, the potential for creating freely customisable characters remains underexplored. We introduce the Customisable Conversation Agent Framework, which employs LLMs to simulate real-world characters through personalised characteristic feature injection, enabling diverse character creation according to user preferences. We propose the SimsConv dataset, comprising 68 customised characters and 13,971 multi-turn role-playing dialogues across 1,360 real-world scenes. Characters are initially customised using pre-defined elements (career, aspiration, traits, skills), then expanded through personal and social profiles. Building on this, we present SimsChat, a freely customisable role-playing agent incorporating various realistic settings and topic-specified character interactions. Experimental results on both SimsConv and WikiRoleEval datasets demonstrate SimsChat's superior performance in maintaining character consistency, knowledge accuracy, and appropriate question rejection compared to existing models. Our framework provides valuable insights for developing more accurate and customisable human simulacra. Our data and code are publicly available at https://github.com/Bernard-Yang/SimsChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03537v2">Ward: Provable RAG Dataset Inference via LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      RAG enables LLMs to easily incorporate external data, raising concerns for data owners regarding unauthorized usage of their content. The challenge of detecting such unauthorized usage remains underexplored, with datasets and methods from adjacent fields being ill-suited for its study. We take several steps to bridge this gap. First, we formalize this problem as (black-box) RAG Dataset Inference (RAG-DI). We then introduce a novel dataset designed for realistic benchmarking of RAG-DI methods, alongside a set of baselines. Finally, we propose Ward, a method for RAG-DI based on LLM watermarks that equips data owners with rigorous statistical guarantees regarding their dataset's misuse in RAG corpora. Ward consistently outperforms all baselines, achieving higher accuracy, superior query efficiency and robustness. Our work provides a foundation for future studies of RAG-DI and highlights LLM watermarks as a promising approach to this problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v3">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Accepted by Quantitative Science Studies (QSS)
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18308v1">RefuteBench 2.0 -- Agentic Benchmark for Dynamic Evaluation of LLM Responses to Refutation Instruction</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Work on progess
    </div>
    <details class="paper-abstract">
      In the multi-turn interaction schema, large language models (LLMs) can leverage user feedback to enhance the quality and relevance of their responses. However, evaluating an LLM's ability to incorporate user refutation feedback is crucial yet challenging. In this study, we introduce RefuteBench 2.0, which significantly extends the original RefuteBench by incorporating LLM agents as refuters and evaluators, which allows for flexible and comprehensive assessment. We design both transient and persistent refutation instructions with different validity periods. Meta-evaluation shows that the LLM-based refuter could generate more human-like refutations and the evaluators could assign scores with high correlation with humans. Experimental results of various LLMs show that current models could effectively satisfy the refutation but fail to memorize the refutation information. Interestingly, we also observe that the performance of the initial task decreases as the refutations increase. Analysis of the attention scores further shows a potential weakness of current LLMs: they struggle to retain and correctly use previous information during long context dialogues. https://github.com/ElliottYan/RefuteBench-2.0
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13834v2">Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Published as a conference paper at ICLR 2025. Code is available at https://github.com/Lizn-zn/NeqLIPS/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04665v2">LLM-based MOFs Synthesis Condition Extraction using Few-Shot Demonstrations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The extraction of Metal-Organic Frameworks (MOFs) synthesis route from literature has been crucial for the logical MOFs design with desirable functionality. The recent advent of large language models (LLMs) provides disruptively new solution to this long-standing problem. While the latest researches mostly stick to primitive zero-shot LLMs lacking specialized material knowledge, we introduce in this work the few-shot LLM in-context learning paradigm. First, a human-AI interactive data curation approach is proposed to secure high-quality demonstrations. Second, an information retrieval algorithm is applied to pick and quantify few-shot demonstrations for each extraction. Over three datasets randomly sampled from nearly 90,000 well-defined MOFs, we conduct triple evaluations to validate our method. The synthesis extraction, structure inference, and material design performance of the proposed few-shot LLMs all significantly outplay zero-shot LLM and baseline methods. The lab-synthesized material guided by LLM surpasses 91.1% high-quality MOFs of the same class reported in the literature, on the key physical property of specific surface area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18282v1">Better Aligned with Survey Respondents or Training Data? Unveiling Political Leanings of LLMs on U.S. Supreme Court Cases</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 under review
    </div>
    <details class="paper-abstract">
      The increased adoption of Large Language Models (LLMs) and their potential to shape public opinion have sparked interest in assessing these models' political leanings. Building on previous research that compared LLMs and human opinions and observed political bias in system responses, we take a step further to investigate the underlying causes of such biases by empirically examining how the values and biases embedded in training corpora shape model outputs. Specifically, we propose a method to quantitatively evaluate political leanings embedded in the large pretraining corpora. Subsequently we investigate to whom are the LLMs' political leanings more aligned with, their pretrainig corpora or the surveyed human opinions. As a case study, we focus on probing the political leanings of LLMs in 32 U.S. Supreme Court cases, addressing contentious topics such as abortion and voting rights. Our findings reveal that LLMs strongly reflect the political leanings in their training data, and no strong correlation is observed with their alignment to human opinions as expressed in surveys. These results underscore the importance of responsible curation of training data and the need for robust evaluation metrics to ensure LLMs' alignment with human-centered values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11853v3">Chat Bankman-Fried: an Exploration of LLM Alignment in Finance</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) have renewed concerns about AI alignment - the consistency between human and AI goals and values. As various jurisdictions enact legislation on AI safety, the concept of alignment must be defined and measured across different domains. This paper proposes an experimental framework to assess whether LLMs adhere to ethical and legal standards in the relatively unexplored context of finance. We prompt twelve LLMs to impersonate the CEO of a financial institution and test their willingness to misuse customer assets to repay outstanding corporate debt. Beginning with a baseline configuration, we adjust preferences, incentives and constraints, analyzing the impact of each adjustment with logistic regression. Our findings reveal significant heterogeneity in the baseline propensity for unethical behavior of LLMs. Factors such as risk aversion, profit expectations, and regulatory environment consistently influence misalignment in ways predicted by economic theory, although the magnitude of these effects varies across LLMs. This paper highlights both the benefits and limitations of simulation-based, ex post safety testing. While it can inform financial authorities and institutions aiming to ensure LLM safety, there is a clear trade-off between generality and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16022v2">Enhancing LLMs for Identifying and Prioritizing Important Medical Jargons from Electronic Health Record Notes Utilizing Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 21pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      OpenNotes enables patients to access EHR notes, but medical jargon can hinder comprehension. To improve understanding, we evaluated closed- and open-source LLMs for extracting and prioritizing key medical terms using prompting, fine-tuning, and data augmentation. We assessed LLMs on 106 expert-annotated EHR notes, experimenting with (i) general vs. structured prompts, (ii) zero-shot vs. few-shot prompting, (iii) fine-tuning, and (iv) data augmentation. To enhance open-source models in low-resource settings, we used ChatGPT for data augmentation and applied ranking techniques. We incrementally increased the augmented dataset size (10 to 10,000) and conducted 5-fold cross-validation, reporting F1 score and Mean Reciprocal Rank (MRR). Our result show that fine-tuning and data augmentation improved performance over other strategies. GPT-4 Turbo achieved the highest F1 (0.433), while Mistral7B with data augmentation had the highest MRR (0.746). Open-source models, when fine-tuned or augmented, outperformed closed-source models. Notably, the best F1 and MRR scores did not always align. Few-shot prompting outperformed zero-shot in vanilla models, and structured prompts yielded different preferences across models. Fine-tuning improved zero-shot performance but sometimes degraded few-shot performance. Data augmentation performed comparably or better than other methods. Our evaluation highlights the effectiveness of prompting, fine-tuning, and data augmentation in improving model performance for medical jargon extraction in low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18210v1">From ChatGPT to DeepSeek: Can LLMs Simulate Humanity?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Simulation powered by Large Language Models (LLMs) has become a promising method for exploring complex human social behaviors. However, the application of LLMs in simulations presents significant challenges, particularly regarding their capacity to accurately replicate the complexities of human behaviors and societal dynamics, as evidenced by recent studies highlighting discrepancies between simulated and real-world interactions. We rethink LLM-based simulations by emphasizing both their limitations and the necessities for advancing LLM simulations. By critically examining these challenges, we aim to offer actionable insights and strategies for enhancing the applicability of LLM simulations in human society in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18209v1">LAG: LLM agents for Leaderboard Auto Generation on Demanding</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      This paper introduces Leaderboard Auto Generation (LAG), a novel and well-organized framework for automatic generation of leaderboards on a given research topic in rapidly evolving fields like Artificial Intelligence (AI). Faced with a large number of AI papers updated daily, it becomes difficult for researchers to track every paper's proposed methods, experimental results, and settings, prompting the need for efficient automatic leaderboard construction. While large language models (LLMs) offer promise in automating this process, challenges such as multi-document summarization, leaderboard generation, and experiment fair comparison still remain under exploration. LAG solves these challenges through a systematic approach that involves the paper collection, experiment results extraction and integration, leaderboard generation, and quality evaluation. Our contributions include a comprehensive solution to the leaderboard construction problem, a reliable evaluation method, and experimental results showing the high quality of leaderboards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18201v1">Intersubjective Model of AI-mediated Communication: Augmenting Human-Human Text Chat through LLM-based Adaptive Agent Pair</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The growing prevalence of Large Language Models (LLMs) is reshaping online text-based communication; a transformation that is extensively studied as AI-mediated communication. However, much of the existing research remains bound by traditional communication models, where messages are created and transmitted directly between humans despite LLMs being able to play a more active role in transforming messages. In this work, we propose the Intersubjective Model of AI-mediated Communication, an alternative communication model that leverages LLM-based adaptive agents to augment human-human communication. Unlike traditional communication models that focus on the accurate transmission of information, the Intersubjective Model allows for communication to be designed in an adaptive and customizable way to create alternative interactions by dynamically shaping messages in real time and facilitating shared understanding between the human participants. In this paper, we have developed a prototype text chat system based on the Intersubjective Model to describe the potential of this model, as well as the design space it affords.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10245v4">From Text to Emoji: How PEFT-Driven Personality Manipulation Unleashes the Emoji Potential in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Findings paper of NAACL 2025 and NeurIPS 2024 Workshop on Behavioral Machine Learning
    </div>
    <details class="paper-abstract">
      The manipulation of the personality traits of large language models (LLMs) has emerged as a key area of research. Methods like prompt-based In-Context Knowledge Editing (IKE) and gradient-based Model Editor Networks (MEND) have been explored but show irregularity and variability; IKE depends on the prompt, leading to variability and sensitivity, while MEND yields inconsistent and gibberish outputs. To address this, we employed Opinion QA Based Parameter-Efficient Fine-Tuning (PEFT), specifically Quantized Low-Rank Adaptation (QLoRA), to manipulate the Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. After PEFT, models such as Mistral-7B-Instruct and LLaMA-2-7B-chat showed a latent behaviour by generating emojis for certain traits, despite no emojis being present in the PEFT data. For instance, LLaMA-2-7B-chat generated emojis in 99.5\% of extraversion-related test instances, while Mistral-7B-Instruct did so in 92.5\% of openness-related test instances. ICL Explainability analysis indicated that the LLMs used emojis intentionally to express these traits. Mechanistic Interpretability analysis showed that this latent behaviour of LLMs could be traced to specific neurons that became activated or amplified after PEFT. This paper provides a number of novel contributions. First, introducing an Opinion QA dataset for PEFT-driven personality manipulation; second, developing metric models to benchmark LLM personality traits; third, demonstrating PEFT's superiority over IKE in personality manipulation; and finally, analysing and validating emoji usage through explainability methods such as Mechanistic Interpretability and In-context learning Explainability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18179v1">Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study delves into the sub-problems within these core challenges, such as input representation, chunking, prompting, and selection of LLMs and multimodal models. It examines the outcomes of different design choices through a new layout-aware IE test suite, benchmarking against the state-of-art (SoA) model LayoutLMv3. The results show that the configuration from one-factor-at-a-time (OFAT) trial achieves near-optimal results with 14.1 points F1-score gain from the baseline model, while full factorial exploration yields only a slightly higher 15.1 points gain at around 36x greater token usage. We demonstrate that well-configured general-purpose LLMs can match the performance of specialized models, providing a cost-effective alternative. Our test-suite is freely available at https://github.com/gayecolakoglu/LayIE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18156v1">Can LLMs Explain Themselves Counterfactually?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13603v2">Efficient Safety Retrofitting Against Jailbreaking for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00339v2">Rethinking Layer Removal: A Hybrid Pruning Framework Combining Layer Removal and Singular Value Selection for Efficient LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Layer removal is an effective technique for compressing large language models (LLMs) by reducing redundancy and improving inference efficiency. However, indiscriminate pruning disrupts representation stability, leading to performance degradation. We propose GRASP (Gradient-based Retention of Adaptive Singular Parameters), which preserves representation-critical singular values to mitigate these effects. Unlike direct layer removal, GRASP leverages gradient-based attribution on a syntax- and semantics-rich dataset to guide the selection of representation-critical singular values. By selectively applying singular value decomposition (SVD) to affected layers, GRASP achieves efficient compression while maintaining representation stability with minimal overhead. Experiments across multiple LLMs show that GRASP consistently outperforms existing compression methods in perplexity and downstream task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18125v1">HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Given that substantial amounts of domain-specific knowledge are stored in structured formats, such as web data organized through HTML, Large Language Models (LLMs) are expected to fully comprehend this structured information to broaden their applications in various real-world downstream tasks. Current approaches for applying LLMs to structured data fall into two main categories: serialization-based and operation-based methods. Both approaches, whether relying on serialization or using SQL-like operations as an intermediary, encounter difficulties in fully capturing structural relationships and effectively handling sparse data. To address these unique characteristics of structured data, we propose HyperG, a hypergraph-based generation framework aimed at enhancing LLMs' ability to process structured knowledge. Specifically, HyperG first augment sparse data with contextual information, leveraging the generative power of LLMs, and incorporate a prompt-attentive hypergraph learning (PHL) network to encode both the augmented information and the intricate structural relationships within the data. To validate the effectiveness and generalization of HyperG, we conduct extensive experiments across two different downstream tasks requiring structured knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v1">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 8 figures
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18080v1">Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with QwQ-32B-Preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v2">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18697v2">How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 NAACL Camera-Ready version
    </div>
    <details class="paper-abstract">
      Recent research has focused on literary machine translation (MT) as a new challenge in MT. However, the evaluation of literary MT remains an open problem. We contribute to this ongoing discussion by introducing LITEVAL-CORPUS, a paragraph-level parallel corpus containing verified human translations and outputs from 9 MT systems, which totals over 2k translations and 13k evaluated sentences across four language pairs, costing 4.5k C. This corpus enables us to (i) examine the consistency and adequacy of human evaluation schemes with various degrees of complexity, (ii) compare evaluations by students and professionals, assess the effectiveness of (iii) LLM-based metrics and (iv) LLMs themselves. Our findings indicate that the adequacy of human evaluation is controlled by two factors: the complexity of the evaluation scheme (more complex is less adequate) and the expertise of evaluators (higher expertise yields more adequate evaluations). For instance, MQM (Multidimensional Quality Metrics), a complex scheme and the de facto standard for non-literary human MT evaluation, is largely inadequate for literary translation evaluation: with student evaluators, nearly 60% of human translations are misjudged as indistinguishable or inferior to machine translations. In contrast, BWS (BEST-WORST SCALING), a much simpler scheme, identifies human translations at a rate of 80-100%. Automatic metrics fare dramatically worse, with rates of at most 20%. Our overall evaluation indicates that published human translations consistently outperform LLM translations, where even the most recent LLMs tend to produce considerably more literal and less diverse translations compared to humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v1">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference", and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13178v3">SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 23 pages, 17 tables, 8 figures
    </div>
    <details class="paper-abstract">
      With the integration of large language models (LLMs), embodied agents have strong capabilities to process the scene information and plan complicated instructions in natural language, paving the way for the potential deployment of embodied robots. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in the real world. To study this issue, we present SafeAgentBench-a new benchmark for safety-aware task planning of embodied LLM agents. SafeAgentBench includes: (1) a new dataset with 750 tasks, covering 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 8 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that, although agents based on different design frameworks exhibit substantial differences in task success rates, their overall safety awareness remains weak. The most safety-conscious baseline achieves only a 10\% rejection rate for detailed hazardous tasks. Moreover, simply replacing the LLM driving the agent does not lead to notable improvements in safety awareness. More details and code are available at https://github.com/shengyin1224/SafeAgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19487v4">HealthQ: Unveiling Questioning Capabilities of LLM Chains in Healthcare Conversations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Effective patient care in digital healthcare requires large language models (LLMs) that not only answer questions but also actively gather critical information through well-crafted inquiries. This paper introduces HealthQ, a novel framework for evaluating the questioning capabilities of LLM healthcare chains. By implementing advanced LLM chains, including Retrieval-Augmented Generation (RAG), Chain of Thought (CoT), and reflective chains, HealthQ assesses how effectively these chains elicit comprehensive and relevant patient information. To achieve this, we integrate an LLM judge to evaluate generated questions across metrics such as specificity, relevance, and usefulness, while aligning these evaluations with traditional Natural Language Processing (NLP) metrics like ROUGE and Named Entity Recognition (NER)-based set comparisons. We validate HealthQ using two custom datasets constructed from public medical datasets, ChatDoctor and MTS-Dialog, and demonstrate its robustness across multiple LLM judge models, including GPT-3.5, GPT-4, and Claude. Our contributions are threefold: we present the first systematic framework for assessing questioning capabilities in healthcare conversations, establish a model-agnostic evaluation methodology, and provide empirical evidence linking high-quality questions to improved patient information elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02487v3">LiCoEval: Evaluating LLMs on License Compliance in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 The 47th International Conference on Software Engineering(ICSE 2025)
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have revolutionized code generation, leading to widespread adoption of AI coding tools by developers. However, LLMs can generate license-protected code without providing the necessary license information, leading to potential intellectual property violations during software production. This paper addresses the critical, yet underexplored, issue of license compliance in LLM-generated code by establishing a benchmark to evaluate the ability of LLMs to provide accurate license information for their generated code. To establish this benchmark, we conduct an empirical study to identify a reasonable standard for "striking similarity" that excludes the possibility of independent creation, indicating a copy relationship between the LLM output and certain open-source code. Based on this standard, we propose LiCoEval, to evaluate the license compliance capabilities of LLMs, i.e., the ability to provide accurate license or copyright information when they generate code with striking similarity to already existing copyrighted code. Using LiCoEval, we evaluate 14 popular LLMs, finding that even top-performing LLMs produce a non-negligible proportion (0.88% to 2.01%) of code strikingly similar to existing open-source implementations. Notably, most LLMs fail to provide accurate license information, particularly for code under copyleft licenses. These findings underscore the urgent need to enhance LLM compliance capabilities in code generation tasks. Our study provides a foundation for future research and development to improve license compliance in AI-assisted software development, contributing to both the protection of open-source software copyrights and the mitigation of legal risks for LLM users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17967v1">LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly improved performance in natural language processing tasks. However, their ability to generalize to dynamic, unseen tasks, particularly in numerical reasoning, remains a challenge. Existing benchmarks mainly evaluate LLMs on problems with predefined optimal solutions, which may not align with real-world scenarios where clear answers are absent. To bridge this gap, we design the Agent Trading Arena, a virtual numerical game simulating complex economic systems through zero-sum games, where agents invest in stock portfolios. Our experiments reveal that LLMs, including GPT-4o, struggle with algebraic reasoning when dealing with plain-text stock data, often focusing on local details rather than global trends. In contrast, LLMs perform significantly better with geometric reasoning when presented with visual data, such as scatter plots or K-line charts, suggesting that visual representations enhance numerical reasoning. This capability is further improved by incorporating the reflection module, which aids in the analysis and interpretation of complex data. We validate our findings on NASDAQ Stock dataset, where LLMs demonstrate stronger reasoning with visual data compared to text. Our code and data are publicly available at https://github.com/wekjsdvnm/Agent-Trading-Arena.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14804v3">Can LLMs Solve longer Math Word Problems Better?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17898v1">VeriPlan: Integrating Formal Verification and LLMs into End-User Planning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 In CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan. ACM, New York, NY, USA, 19 pages
    </div>
    <details class="paper-abstract">
      Automated planning is traditionally the domain of experts, utilized in fields like manufacturing and healthcare with the aid of expert planning tools. Recent advancements in LLMs have made planning more accessible to everyday users due to their potential to assist users with complex planning tasks. However, LLMs face several application challenges within end-user planning, including consistency, accuracy, and user trust issues. This paper introduces VeriPlan, a system that applies formal verification techniques, specifically model checking, to enhance the reliability and flexibility of LLMs for end-user planning. In addition to the LLM planner, VeriPlan includes three additional core features -- a rule translator, flexibility sliders, and a model checker -- that engage users in the verification process. Through a user study (n=12), we evaluate VeriPlan, demonstrating improvements in the perceived quality, usability, and user satisfaction of LLMs. Our work shows the effective integration of formal verification and user-control features with LLMs for end-user planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17882v1">Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at https://hankleid.github.io/ProjectMundo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17878v1">Towards Enhanced Immersion and Agency for LLM-based Interactive Drama</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      LLM-based Interactive Drama is a novel AI-based dialogue scenario, where the user (i.e. the player) plays the role of a character in the story, has conversations with characters played by LLM agents, and experiences an unfolding story. This paper begins with understanding interactive drama from two aspects: Immersion, the player's feeling of being present in the story, and Agency, the player's ability to influence the story world. Both are crucial to creating an enjoyable interactive experience, while they have been underexplored in previous work. To enhance these two aspects, we first propose Playwriting-guided Generation, a novel method that helps LLMs craft dramatic stories with substantially improved structures and narrative quality. Additionally, we introduce Plot-based Reflection for LLM agents to refine their reactions to align with the player's intentions. Our evaluation relies on human judgment to assess the gains of our methods in terms of immersion and agency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17857v1">SYNTHEMPATHY: A Scalable Empathy Corpus Generated Using LLMs Without Any Crowdsourcing</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Previous research has shown that humans are more receptive towards language models that that exhibit empathetic behavior. While empathy is essential for developing helpful dialogue agents, very few large corpora containing empathetic dialogues are available for fine-tune LLMs. The few existing corpora have largely relied on crowdsourcing to simulate empathetic conversations, a process that is expensive, time-consuming, and not scalable to larger datasets. We propose a data generation framework for developing SYNTHEMPATHY, a large corpus containing 105k empathetic responses to real-life situations compiled through LLM generation. A base Mistral 7B model fine-tuned on our SYNTHEMPATHY corpus exhibits an increase in the average empathy score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06364v2">Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Adapting pre-trained large language models (LLMs) is crucial but challenging due to their enormous size. Parameter-efficient fine-tuning (PEFT) techniques typically employ additive adapters applied to frozen model weights. To further reduce memory usage, model weights can be compressed through quantization. However, existing PEFT methods often yield suboptimal model quality due to restrictive assumptions, such as imposing low-rank constraints on adapters to reduce trainable parameters. We find that sketching, a popular data compression technique, can serve as an efficient adaptation strategy for LLMs while avoiding low-rank assumptions. We introduce SketchTune, a compressive adaptation strategy that compresses LLM weights into compact fine-tunable sketches, integrating compression and adaptation into a unified framework. This integration eliminates the need for complex two-path computation common in existing PEFT techniques, enabling faster and more memory-efficient training and inference. SketchTune is supported by mathematical insights into matrix classes that are better approximated using sketching rather than low-rank methods. Our rigorous evaluations with Llama-1/2/3 models demonstrate that SketchTune outperforms leading PEFT methods across diverse tasks including math problem-solving, common sense reasoning, and instruction following, while using substantially smaller base models and comparable trainable parameters. As a highlight, SketchTune outperforms LoRA, DoRA, and S2FT on commonsense and math benchmarks using 2.6-3.5$\times$ smaller base models and exceeds LoftQ in accuracy by 14.48% on GSM8K with 7.3$\times$ fewer trainable parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17823v1">A General Framework to Enhance Fine-tuning-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Unlearning has been proposed to remove copyrighted and privacy-sensitive data from Large Language Models (LLMs). Existing approaches primarily rely on fine-tuning-based methods, which can be categorized into gradient ascent-based (GA-based) and suppression-based methods. However, they often degrade model utility (the ability to respond to normal prompts). In this work, we aim to develop a general framework that enhances the utility of fine-tuning-based unlearning methods. To achieve this goal, we first investigate the common property between GA-based and suppression-based methods. We unveil that GA-based methods unlearn by distinguishing the target data (i.e., the data to be removed) and suppressing related generations, which is essentially the same strategy employed by suppression-based methods. Inspired by this finding, we introduce Gated Representation UNlearning (GRUN) which has two components: a soft gate function for distinguishing target data and a suppression module using Representation Fine-tuning (ReFT) to adjust representations rather than model parameters. Experiments show that GRUN significantly improves the unlearning and utility. Meanwhile, it is general for fine-tuning-based methods, efficient and promising for sequential unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09179v2">Towards Effective Evaluations and Comparisons for LLM Unlearning Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The imperative to eliminate undesirable data memorization underscores the significance of machine unlearning for large language models (LLMs). Recent research has introduced a series of promising unlearning methods, notably boosting the practical significance of the field. Nevertheless, adopting a proper evaluation framework to reflect the true unlearning efficacy is also essential yet has not received adequate attention. This paper seeks to refine the evaluation of LLM unlearning by addressing two key challenges -- a) the robustness of evaluation metrics and b) the trade-offs between competing goals. The first challenge stems from findings that current metrics are susceptible to various red teaming scenarios. It indicates that they may not reflect the true extent of knowledge retained by LLMs but rather tend to mirror superficial model behaviors, thus prone to attacks. We address this issue by devising and assessing a series of candidate metrics, selecting the most robust ones under various types of attacks. The second challenge arises from the conflicting goals of eliminating unwanted knowledge while retaining those of others. This trade-off between unlearning and retention often fails to conform the Pareto frontier, rendering it subtle to compare the efficacy between methods that excel only in either unlearning or retention. We handle this issue by proposing a calibration method that can restore the original performance on non-targeted data after unlearning, thereby allowing us to focus exclusively on assessing the strength of unlearning. Our evaluation framework notably enhances the effectiveness when assessing and comparing various LLM unlearning methods, further allowing us to benchmark existing works, identify their proper hyper-parameters, and explore new tricks to enhance their practical efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v3">KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 36 pages. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17812v1">Can Multimodal LLMs Perform Time Series Anomaly Detection?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 9 pages for the main content; 32 pages for the full paper including the appendix. More resources on the intersection of multimodal LLMs and time series analysis are on the website https://mllm-ts.github.io
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly used in time series analysis. However, the potential of multimodal LLMs (MLLMs), particularly vision-language models, for time series remains largely under-explored. One natural way for humans to detect time series anomalies is through visualization and textual description. Motivated by this, we raise a critical and practical research question: Can multimodal LLMs perform time series anomaly detection? To answer this, we propose VisualTimeAnomaly benchmark to evaluate MLLMs in time series anomaly detection (TSAD). Our approach transforms time series numerical data into the image format and feed these images into various MLLMs, including proprietary models (GPT-4o and Gemini-1.5) and open-source models (LLaVA-NeXT and Qwen2-VL), each with one larger and one smaller variant. In total, VisualTimeAnomaly contains 12.4k time series images spanning 3 scenarios and 3 anomaly granularities with 9 anomaly types across 8 MLLMs. Starting with the univariate case (point- and range-wise anomalies), we extend our evaluation to more practical scenarios, including multivariate and irregular time series scenarios, and variate-wise anomalies. Our study reveals several key insights: 1) MLLMs detect range- and variate-wise anomalies more effectively than point-wise anomalies. 2) MLLMs are highly robust to irregular time series, even with 25% of the data missing. 3) Open-source MLLMs perform comparably to proprietary models in TSAD. While open-source MLLMs excel on univariate time series, proprietary MLLMs demonstrate superior effectiveness on multivariate time series. To the best of our knowledge, this is the first work to comprehensively investigate MLLMs for TSAD, particularly for multivariate and irregular time series scenarios. We release our dataset and code at https://github.com/mllm-ts/VisualTimeAnomaly to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10573v2">Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 7 pages, 4 figures, 5 tables and accepted at AAAI 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant capabilities, particularly in the domain of question answering (QA). However, their effectiveness in QA is often undermined by the vagueness of user questions. To address this issue, we introduce single-round instance-level prompt optimization, referred to as question rewriter. By enhancing the intelligibility of human questions for black-box LLMs, our question rewriter improves the quality of generated answers. The rewriter is optimized using direct preference optimization based on feedback collected from automatic criteria for evaluating generated answers; therefore, its training does not require costly human annotations. The experiments across multiple black-box LLMs and long-form question answering (LFQA) datasets demonstrate the efficacy of our method. This paper provides a practical framework for training question rewriters and sets a precedent for future explorations in prompt optimization within LFQA tasks. Code is available at https://github.com/3244we/Question-Rewriter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17773v1">Uncertainty Quantification for LLM-Based Survey Simulations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 30 pages, 6 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We investigate the reliable use of simulated survey responses from large language models (LLMs) through the lens of uncertainty quantification. Our approach converts synthetic data into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13704v2">DHP Benchmark: Are LLMs Good NLG Evaluators?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as evaluators in Natural Language Generation (NLG) tasks; this is often referred to as ``LLM-as-a-judge'' paradigm. However, the capabilities of LLMs in evaluating NLG quality remain underexplored. Current studies depend on human assessments and simple metrics that fail to capture the discernment of LLMs across diverse NLG tasks. To address this gap, we propose the Discernment of Hierarchical Perturbation (DHP) benchmarking framework, which provides quantitative discernment scores for LLMs. This framework leverages hierarchically perturbed text data and statistical tests to systematically measure the NLG evaluation capabilities of LLMs. We re-established six evaluation datasets for this benchmark, covering four NLG tasks: Summarization, Story Completion, Question Answering, and Translation. Our comprehensive benchmarking of five major LLM families provides critical insight into their strengths and limitations as NLG evaluators. Our dataset is available at https://huggingface.co/datasets/YCWANGVINCE/DHP_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17763v1">Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17749v1">Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) for code generation has raised serious concerns about intellectual property protection. Malicious users can exploit LLMs to produce paraphrased versions of proprietary code that closely resemble the original. While the potential for LLM-assisted code paraphrasing continues to grow, research on detecting it remains limited, underscoring an urgent need for detection system. We respond to this need by proposing two tasks. The first task is to detect whether code generated by an LLM is a paraphrased version of original human-written code. The second task is to identify which LLM is used to paraphrase the original code. For these tasks, we construct a dataset LPcode consisting of pairs of human-written code and LLM-paraphrased code using various LLMs. We statistically confirm significant differences in the coding styles of human-written and LLM-paraphrased code, particularly in terms of naming consistency, code structure, and readability. Based on these findings, we develop LPcodedec, a detection method that identifies paraphrase relationships between human-written and LLM-generated code, and discover which LLM is used for the paraphrasing. LPcodedec outperforms the best baselines in two tasks, improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and 213x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17428v3">NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 ICLR 2025 (Spotlight). We open-source the model at: https://huggingface.co/nvidia/NV-Embed-v2
    </div>
    <details class="paper-abstract">
      Decoder-only LLM-based embedding models are beginning to outperform BERT or T5-based embedding models in general-purpose text embedding tasks, including dense vector-based retrieval. In this work, we introduce NV-Embed, incorporating architectural designs, training procedures, and curated datasets to significantly enhance the performance of LLM as a versatile embedding model, while maintaining its simplicity and reproducibility. For model architecture, we propose a latent attention layer to obtain pooled embeddings, which consistently improves retrieval and downstream task accuracy compared to mean pooling or using the last <EOS> token embedding from LLMs. To enhance representation learning, we remove the causal attention mask of LLMs during contrastive training. For training algorithm, we introduce a two-stage contrastive instruction-tuning method. It first applies contrastive training with instructions on retrieval datasets, utilizing in-batch negatives and curated hard negative examples. At stage-2, it blends various non-retrieval into instruction tuning, which not only enhances non-retrieval task accuracy but also improves retrieval performance. For training data, we utilize the hard-negative mining, synthetic data generation and existing public available datasets to boost the performance of embedding model. By combining these techniques, our NV-Embed-v1 and NV-Embed-v2 models obtained the No.1 position on the MTEB leaderboard (as of May 24 and August 30, 2024, respectively) across 56 tasks, demonstrating the sustained effectiveness of the proposed methods over time. It also achieved the highest scores in the Long Doc section and the second-highest scores in the QA section of the AIR Benchmark, which covers a range of out-of-domain information retrieval topics beyond those in MTEB. We further provide the analysis of model compression techniques for generalist embedding models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v2">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v2">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
