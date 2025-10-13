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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09599v1">Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Our code and data are available at https://github.com/VILA-Lab/PTTS
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v3">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 82 pages, 28 figures, 32 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v5">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo (Fine-grained Semantic Comparison), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSCo more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20179v5">Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Conference on Language Modeling (COLM) 2025, Project site: https://amrl.cs.utexas.edu/robo-instruct/
    </div>
    <details class="paper-abstract">
      Code LLMs have shown promising results with converting tasks in natural language to programs that can be executed by service robots. We are interested in finetuning small, specialized LLMs for this purpose, but collecting datasets of task-program pairs specific to each robot is time-consuming and expensive. While approaches such as SELF-INSTRUCT and EVOL-INSTRUCT are capable of generating novel tasks given a few examples, they are unable to provide the corresponding programs that correctly abide by physical-world and robot-constraints using the provided programming interface. Using a simulator is a natural potential solution to checking for such constraints, but building simulation environments that can handle arbitrary tasks and their necessary objects and locations, is challenging. To address these challenges, we introduce ROBO-INSTRUCT, which synthesizes task-specific simulation environments on the fly during program execution, by opportunistically inferring entity properties and enforcing corresponding constraints based on how the entities are used in the task program. Additionally, ROBO-INSTRUCT integrates an LLM-aided post-processing procedure to refine instructions for better alignment with robot programs. We demonstrate the effectiveness of ROBO-INSTRUCT across multiple LLMs, showing that our fine-tuned models outperform all baseline methods and even match or surpass the performance of several larger and proprietary models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09471v1">Getting Your Indices in a Row: Full-Text Search for LLM Training Data for Real World</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) is determined by their training data. Despite the proliferation of open-weight LLMs, access to LLM training data has remained limited. Even for fully open LLMs, the scale of the data makes it all but inscrutable to the general scientific community, despite potentially containing critical data scraped from the internet. In this paper, we present the full-text indexing pipeline for the Apertus LLM training data. Leveraging Elasticsearch parallel indices and the Alps infrastructure, a state-of-the-art, highly energy-efficient arm64 supercluster, we were able to index 8.6T tokens out of 15.2T used to train the Apertus LLM family, creating both a critical LLM safety tool and effectively an offline, curated, open web search engine. Our contribution is threefold. First, we demonstrate that Elasticsearch can be successfully ported onto next-generation arm64-based infrastructure. Second, we demonstrate that full-text indexing at the scale of modern LLM training datasets and the entire open web is feasible and accessible. Finally, we demonstrate that such indices can be used to ensure previously inaccessible jailbreak-agnostic LLM safety. We hope that our findings will be useful to other teams attempting large-scale data indexing and facilitate the general transition towards greener computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07180v2">Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the video domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. Furthermore, we propose two potential training-free mitigation strategies, revealing potential paths for reducing sycophantic bias: (i) enhancing visual grounding through interpretable key-frame selection and (ii) steering model behavior away from sycophancy via targeted, inference-time intervention on its internal neural representations. Our code is available at https://github.com/William030422/Video-Sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2208.08067v2">K-ASTRO: Structure-Aware Adaptation of LLMs for Code Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming software engineering tasks, including code vulnerability detection-a critical area of software security. However, existing methods often rely on resource-intensive models or graph-based techniques, limiting their accessibility and practicality. This paper introduces K-ASTRO, a lightweight Transformer model that combines semantic embeddings from LLMs with structural features of Abstract Syntax Trees (ASTs) to improve both efficiency and accuracy in code vulnerability detection. Our approach introduces an AST-based augmentation technique inspired by mutation testing, a structure-aware attention mechanism that incorporates augmented AST features, and a joint adaptation pipeline to unify code semantics and syntax. Experimental results on three large-scale datasets, including BigVul, DiverseVul, and PrimeVul-demonstrate state-of-the-art performance while enabling rapid inference on CPUs with minimal training time. By offering a scalable, interpretable, and efficient solution, K-ASTRO bridges the gap between LLM advancements and practical software vulnerability detection, providing open-sourced tools to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09424v1">The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      This paper presents a comparative study of context management strategies for end-to-end Spoken Dialog State Tracking using Speech-LLMs. We systematically evaluate traditional multimodal context (combining text history and spoken current turn), full spoken history, and compressed spoken history approaches. Our experiments on the SpokenWOZ corpus demonstrate that providing the full spoken conversation as input yields the highest performance among models of similar size, significantly surpassing prior methods. Furthermore, we show that attention-pooling-based compression of the spoken history offers a strong trade-off, maintaining competitive accuracy with reduced context size. Detailed analysis confirms that improvements stem from more effective context utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08604v2">Preference Discerning with LLM-Enhanced Generative Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted at TMLR, Code available at https://github.com/facebookresearch/preference_discerning
    </div>
    <details class="paper-abstract">
      In sequential recommendation, models recommend items based on user's interaction history. To this end, current models usually incorporate information such as item descriptions and user intent or preferences. User preferences are usually not explicitly given in open-source datasets, and thus need to be approximated, for example via large language models (LLMs). Current approaches leverage approximated user preferences only during training and rely solely on the past interaction history for recommendations, limiting their ability to dynamically adapt to changing preferences, potentially reinforcing echo chambers. To address this issue, we propose a new paradigm, namely preference discerning, which explicitly conditions a generative recommendation model on user preferences in natural language within its context. To evaluate preference discerning, we introduce a novel benchmark that provides a holistic evaluation across various scenarios, including preference steering and sentiment following. Upon evaluating current state-of-the-art methods on our benchmark, we discover that their ability to dynamically adapt to evolving user preferences is limited. To address this, we propose a new method named Mender ($\textbf{M}$ultimodal Prefer$\textbf{en}$ce $\textbf{D}$iscern$\textbf{er}$), which achieves state-of-the-art performance in our benchmark. Our results show that Mender effectively adapts its recommendation guided by human preferences, even if not observed during training, paving the way toward more flexible recommendation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09400v1">TIT: A Tree-Structured Instruction Tuning Approach for LLM-Based Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong performance in automated source-to-target code translation through pretraining on extensive code corpora. However, mainstream LLM-based code translation methods suffer from two critical limitations. First, they are highly sensitive to language-specific features, which often introduce source-language syntax or lexicon into the output, leading to syntactic confusion. Second, they lack fine-grained semantic alignment due to an over-reliance on function-level parallel datasets, resulting in semantic misalignment between the translated code and the original source. To overcome these limitations, we propose TIT, a Tree-structured Instruction Tuning paradigm for LLM-based code translation. Specifically, TIT consists of three modules. First, to mitigate syntactic confusion, the syntactic information representation module integrates language-agnostic syntactic features via structured parsing. Then, to generate high-quality fine-grained parallel data, the fine-grained parallel dataset augmentation module aligns nodes with code segments through statement-level segmentation and contrastive matching. Finally, we leverage the dual-stage tree instruction tuning module to alleviate the contextual processing burden on the LLM caused by the introduction of syntactic information. The first stage employs syntax-aware fine-tuning to enable the LLM to autonomously comprehend structured syntactic information, while the second stage utilizes code generation fine-tuning to guide the model in generating accurate target code based on function-level syntactic dependencies. The experimental results demonstrate that the proposed method significantly outperforms existing approaches in multiple LLMs, achieving a success rate 1.22x-1.75x higher in code translation while markedly reducing syntactic confusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09393v1">ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender systems.Existing approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity users.To address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity users.With a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce platform.ChoirRec improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09378v1">The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Recent efforts to accelerate LLM pretraining have focused on computationally-efficient approximations that exploit second-order structure. This raises a key question for large-scale training: how much performance is forfeited by these approximations? To probe this question, we establish a practical upper bound on iteration complexity by applying full Gauss-Newton (GN) preconditioning to transformer models of up to 150M parameters. Our experiments show that full GN updates yield substantial gains over existing optimizers, achieving a 5.4x reduction in training iterations compared to strong baselines like SOAP and Muon. Furthermore, we find that a precise layerwise GN preconditioner, which ignores cross-layer information, nearly matches the performance of the full GN method. Collectively, our results suggest: (1) the GN approximation is highly effective for preconditioning, implying higher-order loss terms may not be critical for convergence speed; (2) the layerwise Hessian structure contains sufficient information to achieve most of these potential gains; and (3) a significant performance gap exists between current approximate methods and an idealized layerwise oracle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09359v1">Understanding the Effects of Domain Finetuning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuned for specific domains exhibit strong performance; however, the underlying mechanisms by which this fine-tuning reshapes their parametric space are not well understood. Prior works primarily focus on auto-regressive or general-purpose instruct models, leaving domain-specialised LLMs under-explored. We present the first systematic study of domain-specific fine-tuning in large medical language models. Our analysis reveals that fine-tuning modifies only a small subset of the representational subspace, essentially preserving the pre-trained model's representation. To interpret these changes in subspaces, we propose tuning vectors, a novel framework inspired by task vectors, which explicitly capture the directional parameter shifts induced by fine-tuning. We demonstrate that these vectors are critical for enhancing both instruction-following and generation quality. Furthermore, combining tuning vectors across different domains yields improved generalisation. Upon closer inspection of directional alignment, we find these vectors primarily write new directional information into the MLP layers of the model, while amplifying existing directions in attention heads. Our findings offer new insights into LLM adaptation and provide a general, interpretable framework for analysing specialisation in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06944v3">AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 The paper is currently under investigation regarding concerns of potential academic misconduct. While the investigation is ongoing, the authors have voluntarily requested to withdraw the manuscript
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically fine-tuned for reasoning tasks through a two-stage pipeline of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL), a process fraught with catastrophic forgetting and suboptimal trade-offs between imitation and exploration. Recent single-stage methods attempt to unify SFT and RL using heuristics, but lack a principled mechanism for dynamically balancing the two paradigms. In this paper, we reframe this challenge through the theoretical lens of \textbf{implicit rewards}, viewing SFT and RL not as distinct methods but as complementary reward signals. We introduce \textbf{Adaptive Meta Fine-Tuning (AMFT)}, a novel single-stage algorithm that learns the optimal balance between SFT's implicit, path-level reward and RL's explicit, outcome-based reward. The core of AMFT is a \textbf{meta-gradient adaptive weight controller} that treats the SFT-RL balance as a learnable parameter, dynamically optimizing it to maximize long-term task performance. This forward-looking approach, regularized by policy entropy for stability, autonomously discovers an effective training curriculum. We conduct a comprehensive evaluation on challenging benchmarks spanning mathematical reasoning, abstract visual reasoning (General Points), and vision-language navigation (V-IRL). AMFT consistently establishes a new state-of-the-art and demonstrats superior generalization on out-of-distribution (OOD) tasks. Ablation studies and training dynamic analysis confirm that the meta-learning controller is crucial for AMFT's stability, sample efficiency, and performance, offering a more principled and effective paradigm for LLM alignment. Our codes are open-sourced via https://github.com/hlxtsyj/AMFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09347v1">LLP: LLM-based Product Pricing in E-commerce</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Unlike Business-to-Consumer e-commerce platforms (e.g., Amazon), inexperienced individual sellers on Consumer-to-Consumer platforms (e.g., eBay) often face significant challenges in setting prices for their second-hand products efficiently. Therefore, numerous studies have been proposed for automating price prediction. However, most of them are based on static regression models, which suffer from poor generalization performance and fail to capture market dynamics (e.g., the price of a used iPhone decreases over time). Inspired by recent breakthroughs in Large Language Models (LLMs), we introduce LLP, the first LLM-based generative framework for second-hand product pricing. LLP first retrieves similar products to better align with the dynamic market change. Afterwards, it leverages the LLMs' nuanced understanding of key pricing information in free-form text to generate accurate price suggestions. To strengthen the LLMs' domain reasoning over retrieved products, we apply a two-stage optimization, supervised fine-tuning (SFT) followed by group relative policy optimization (GRPO), on a dataset built via bidirectional reasoning. Moreover, LLP employs a confidence-based filtering mechanism to reject unreliable price suggestions. Extensive experiments demonstrate that LLP substantially surpasses existing methods while generalizing well to unseen categories. We have successfully deployed LLP on Xianyu\footnote\{Xianyu is China's largest second-hand e-commerce platform.\}, significantly outperforming the previous pricing method. Under the same 30\% product coverage, it raises the static adoption rate (SAR) from 40\% to 72\%, and maintains a strong SAR of 47\% even at 90\% recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09332v1">FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLM) have achieved remarkable performance, their enormous parameter counts hinder deployment on resource-constrained hardware. Low-rank compression can reduce both memory usage and computational demand, but applying a uniform compression ratio across all layers often leads to significant performance degradation, and previous methods perform poorly during decoding. To address these issues, we propose the Fine-grained Low-Rank Compressor (FLRC), which efficiently determines an optimal rank allocation for each layer, and incorporates progressive low-rank decoding to maintain text generation quality. Comprehensive experiments on diverse benchmarks demonstrate the superiority of FLRC, achieving up to a 17% improvement in ROUGE-L on summarization tasks compared to state-of-the-art low-rank compression methods, establishing a more robust and efficient framework to improve LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10706v2">GestureCoach: Rehearsing for Engaging Talks with LLM-Driven Gesture Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted at UIST 2025
    </div>
    <details class="paper-abstract">
      This paper introduces GestureCoach, a system designed to help speakers deliver more engaging talks by guiding them to gesture effectively during rehearsal. GestureCoach combines an LLM-driven gesture recommendation model with a rehearsal interface that proactively cues speakers to gesture appropriately. Trained on experts' gesturing patterns from TED talks, the model consists of two modules: an emphasis proposal module, which predicts when to gesture by identifying gesture-worthy text segments in the presenter notes, and a gesture identification module, which determines what gesture to use by retrieving semantically appropriate gestures from a curated gesture database. Results of a model performance evaluation and user study (N=30) show that the emphasis proposal module outperforms off-the-shelf LLMs in identifying suitable gesture regions, and that participants rated the majority of these predicted regions and their corresponding gestures as highly appropriate. A subsequent user study (N=10) showed that rehearsing with GestureCoach encouraged speakers to gesture and significantly increased gesture diversity, resulting in more engaging talks. We conclude with design implications for future AI-driven rehearsal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22424v3">Issue Localization via LLM-Driven Iterative Code Graph Searching</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      Issue solving aims to generate patches to fix reported issues in real-world code repositories according to issue descriptions. Issue localization forms the basis for accurate issue solving. Recently, LLM-based issue localization methods have demonstrated state-of-the-art performance. However, these methods either search from files mentioned in issue descriptions or in the whole repository and struggle to balance the breadth and depth of the search space to converge on the target efficiently. Moreover, they allow LLM to explore whole repositories freely, making it challenging to control the search direction to prevent the LLM from searching for incorrect targets. This paper introduces CoSIL, an LLM-driven, powerful function-level issue localization method without training or indexing. CoSIL employs a two-phase code graph search strategy. It first conducts broad exploration at the file level using dynamically constructed module call graphs, and then performs in-depth analysis at the function level by expanding the module call graph into a function call graph and executing iterative searches. To precisely control the search direction, CoSIL designs a pruner to filter unrelated directions and irrelevant contexts. To avoid incorrect interaction formats in long contexts, CoSIL introduces a reflection mechanism that uses additional independent queries in short contexts to enhance formatted abilities. Experiment results demonstrate that CoSIL achieves a Top-1 localization accuracy of 43.3\% and 44.6\% on SWE-bench Lite and SWE-bench Verified, respectively, with Qwen2.5-Coder-32B, average outperforming the state-of-the-art methods by 96.04\%. When CoSIL is integrated into an issue-solving method, Agentless, the issue resolution rate improves by 2.98\%--30.5\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09244v1">Fundamentals of Building Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      This paper reviews the architecture and implementation methods of agents powered by large language models (LLMs). Motivated by the limitations of traditional LLMs in real-world tasks, the research aims to explore patterns to develop "agentic" LLMs that can automate complex tasks and bridge the performance gap with human capabilities. Key components include a perception system that converts environmental percepts into meaningful representations; a reasoning system that formulates plans, adapts to feedback, and evaluates actions through different techniques like Chain-of-Thought and Tree-of-Thought; a memory system that retains knowledge through both short-term and long-term mechanisms; and an execution system that translates internal decisions into concrete actions. This paper shows how integrating these systems leads to more capable and generalized software bots that mimic human cognitive processes for autonomous and intelligent behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09227v1">RegexPSPACE: A Benchmark for Evaluating LLM Reasoning on PSPACE-complete Regex Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong performance across natural language processing (NLP), mathematical reasoning, and programming, and recent large reasoning models (LRMs) further emphasize explicit reasoning. Yet their computational limits, particularly spatial complexity constrained by finite context windows, remain poorly understood. While recent works often focus on problems within the NP complexity class, we push the boundary by introducing a novel benchmark grounded in two PSPACE-complete regular expression (regex) problems: equivalence decision (RegexEQ) and minimization (RegexMin). PSPACE-complete problems serve as a more rigorous standard for assessing computational capacity, as their solutions require massive search space exploration. We perform a double-exponential space exploration to construct a labeled dataset of over a million regex instances with a sound filtering process to build the benchmark. We conduct extensive evaluations on 6 LLMs and 5 LRMs of varying scales, revealing common failure patterns such as verbosity and repetition. With its well-defined structure and quantitative evaluation metrics, this work presents the first empirical investigation into the spatial computational limitations of LLMs and LRMs, offering a new framework for evaluating their advanced reasoning capabilities. Our code is available at https://github.com/hyundong98/RegexPSPACE .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20854v2">An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and other automated techniques have been increasingly used to support software developers by generating software artifacts such as code snippets, patches, and comments. However, accurately assessing the correctness of these generated artifacts remains a significant challenge. On one hand, human evaluation provides high accuracy but is labor-intensive and lacks scalability. On the other hand, many automatic evaluation metrics are scalable and require minimal human effort, but they often fail to accurately reflect the actual correctness of generated software artifacts. In this paper, we present SE-Jury, the first evaluation metric for LLM-as-Ensemble-Judge specifically designed to accurately assess the correctness of generated software artifacts. SE-Jury first defines five distinct evaluation strategies, each implemented by an independent judge. A dynamic team selection mechanism then identifies the most appropriate subset of judges as a team to produce a final correctness score through ensembling. We evaluate SE-Jury across a diverse set of software engineering (SE) benchmarks that span three popular SE tasks: code generation, automated program repair, and code summarization. Results demonstrate that SE-Jury consistently achieves a higher correlation with human judgments, with improvements ranging from 29.6% to 140.8% over existing automatic metrics. SE-Jury reaches agreement levels with human annotators that are close to inter-annotator agreement in code generation and program repair. These findings underscore SE-Jury's potential as a scalable and reliable alternative to human evaluation in these SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19114v2">Understanding and Improving Information Preservation in Prompt Compression for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted to EMNLP 2025 (Findings), 22 pages, 6 figures, 24 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Using our framework, we analyze state-of-the-art soft and hard compression methods and show that some fail to preserve key details from the original prompt, limiting performance on complex tasks. By identifying these limitations, we are able to improve one soft prompting method by controlling compression granularity, achieving up to +23% in downstream performance, +8 BERTScore points in grounding, and 2.7x more entities preserved in compression. Ultimately, we find that the best effectiveness/compression rate trade-off is achieved with soft prompting combined with sequence-level training.The code is available at https://github.com/amazon-science/information-preservation-in-prompt-compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09211v1">DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01694v3">Student-AI Interaction in an LLM-Empowered Learning Environment: A Cluster Analysis of Engagement Profiles</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 15 pages, 10 figures; Reported on ECER 2025
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into educational practice enables personalized learning by accommodating diverse learner behaviors. This study explored diverse learner profiles within a multi-agent, LLM-empowered learning environment. Data was collected from 312 undergraduate students at a university in China as they participated in a six-module course. Based on hierarchical cluster analyses of system profiles and student-AI interactive dialogues, we found that students exhibit varied behavioral, cognitive, and emotional engagement tendencies. This analysis allowed us to identify two types of dropouts (early dropouts and stagnating interactors) and three completer profiles (active questioners, responsive navigators, and lurkers). The results showed that high levels of interaction do not always equate to productive learning and vice versa. Prior knowledge significantly influenced interaction patterns and short-term learning benefits. Further analysis of the human-AI dialogues revealed that some students actively engaged in knowledge construction, while others displayed a high frequency of regulatory behaviors. Notably, both groups of students achieved comparable learning gains, demonstrating the effectiveness of the multi-agent learning environment in supporting personalized learning. These results underscore the complex and multifaceted nature of engagement in human-AI collaborative learning and provide practical implications for the design of adaptive educational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16293v3">Robust LLM Training Infrastructure at ByteDance</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The training scale of large language models (LLMs) has reached tens of thousands of GPUs and is still continuously expanding, enabling faster learning of larger models. Accompanying the expansion of the resource scale is the prevalence of failures (CUDA error, NaN values, job hang, etc.), which poses significant challenges to training stability. Any large-scale LLM training infrastructure should strive for minimal training interruption, efficient fault diagnosis, and effective failure tolerance to enable highly efficient continuous training. This paper presents ByteRobust, a large-scale GPU infrastructure management system tailored for robust and stable training of LLMs. It exploits the uniqueness of LLM training process and gives top priorities to detecting and recovering failures in a routine manner. Leveraging parallelisms and characteristics of LLM training, ByteRobust enables high-capacity fault tolerance, prompt fault demarcation, and localization with an effective data-driven approach, comprehensively ensuring continuous and efficient training of LLM tasks. ByteRobust is deployed on a production GPU platform with over 200,000 GPUs and achieves 97% ETTR for a three-month training job on 9,600 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01752v2">Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, exposing gradients during training can leak sensitive information about the underlying data, raising privacy and security concerns such as susceptibility to data poisoning attacks. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide non-vacuous generalization bounds and strong theoretical guarantees for differential privacy, robustness to data poisoning attacks, and extraction attacks. In experiments with LLMs, we demonstrate empirically that black-box optimization methods-despite the scalability and computational challenges inherent to black-box approaches-are able to learn, showing how a few iterations of BBoxER improve performance, generalize well on a benchmark of reasoning datasets, and are robust to membership inference attacks. This positions BBoxER as an attractive add-on on top of gradient-based optimization, offering suitability for deployment in restricted or privacy-sensitive environments while also providing non-vacuous generalization guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09158v1">Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality Traits by LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 8 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This study proposes augmenting dialog data with think-aloud utterances (TAUs) for modeling individual personalities in text chat by LLM. TAU is a verbalization of a speaker's thought before articulating the utterance. We expect "persona LLMs" trained with TAU-augmented data can mimic the speaker's personality trait better. We tested whether the trained persona LLMs obtain the human personality with respect to Big Five, a framework characterizing human personality traits from five aspects. The results showed that LLMs trained with TAU-augmented data more closely align to the speakers' Agreeableness and Neuroticism of Big Five than those trained with original dialog data. We also found that the quality of TAU-augmentation impacts persona LLM's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07985v2">Fewer Weights, More Problems: A Practical Attack on LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09106v1">When Retrieval Succeeds and Fails: Rethinking Retrieval-Augmented Generation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled a wide range of applications through their powerful capabilities in language understanding and generation. However, as LLMs are trained on static corpora, they face difficulties in addressing rapidly evolving information or domain-specific queries. Retrieval-Augmented Generation (RAG) was developed to overcome this limitation by integrating LLMs with external retrieval mechanisms, allowing them to access up-to-date and contextually relevant knowledge. However, as LLMs themselves continue to advance in scale and capability, the relative advantages of traditional RAG frameworks have become less pronounced and necessary. Here, we present a comprehensive review of RAG, beginning with its overarching objectives and core components. We then analyze the key challenges within RAG, highlighting critical weakness that may limit its effectiveness. Finally, we showcase applications where LLMs alone perform inadequately, but where RAG, when combined with LLMs, can substantially enhance their effectiveness. We hope this work will encourage researchers to reconsider the role of RAG and inspire the development of next-generation RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09058v1">Model-Assisted and Human-Guided: Perceptions and Practices of Software Professionals Using LLMs for Coding</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models have quickly become a central component of modern software development workflows, and software practitioners are increasingly integrating LLMs into various stages of the software development lifecycle. Despite the growing presence of LLMs, there is still a limited understanding of how these tools are actually used in practice and how professionals perceive their benefits and limitations. This paper presents preliminary findings from a global survey of 131 software practitioners. Our results reveal how LLMs are utilized for various coding-specific tasks. Software professionals report benefits such as increased productivity, reduced cognitive load, and faster learning, but also raise concerns about LLMs' inaccurate outputs, limited context awareness, and associated ethical risks. Most developers treat LLMs as assistive tools rather than standalone solutions, reflecting a cautious yet practical approach to their integration. Our findings provide an early, practitioner-focused perspective on LLM adoption, highlighting key considerations for future research and responsible use in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11593v3">Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 This manuscript has been accepted for publication in Springer Neural Computing and Applications
    </div>
    <details class="paper-abstract">
      State-of-The-Art (SoTA) image captioning models are often trained on the MicroSoft Common Objects in Context (MS-COCO) dataset, which contains human-annotated captions with an average length of approximately ten tokens. Although effective for general scene understanding, these short captions often fail to capture complex scenes and convey detailed information. Moreover, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects, thus overlooking finer details. In this paper, we present a novel approach to generate richer and more informative image captions by combining the captions generated from different SoTA captioning models. Our proposed method requires no additional model training: given an image, it leverages pre-trained models from the literature to generate the initial captions, and then ranks them using a newly introduced image-text-based metric, which we name BLIPScore. Subsequently, the top two captions are fused using a Large Language Model (LLM) to produce the final, more detailed description. Experimental results on the MS-COCO and Flickr30k test sets demonstrate the effectiveness of our approach in terms of caption-image alignment and hallucination reduction according to the ALOHa, CAPTURE, and Polos metrics. A subjective study lends additional support to these results, suggesting that the captions produced by our model are generally perceived as more consistent with human judgment. By combining the strengths of diverse SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich and informative nature of human-generated descriptions. This advance enables the generation of more suitable captions for the training of both vision-language and captioning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09045v1">Cost-Efficient Long Code Translation using LLMs while Leveraging Identifier Replacements</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      In the domain of software development, LLMs have been utilized to automate tasks such as code translation, where source code from one programming language is translated to another while preserving its functionality. However, LLMs often struggle with long source codes that don't fit into the context window, which produces inaccurate translations. To address this, we propose a novel zero-shot code translation method that incorporates identifier replacement. By substituting user-given long identifiers with generalized placeholders during translation, our method allows the LLM to focus on the logical structure of the code, by reducing token count and memory usage, which improves the efficiency and cost-effectiveness of long code translation. Our empirical results demonstrate that our approach preserves syntactical and hierarchical information and produces translation results with reduced tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09007v1">LLM Unlearning on Noisy Forget Sets: A Study of Incomplete, Rewritten, and Watermarked Data</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by 18th ACM Workshop on Artificial Intelligence and Security (AISec'25)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable generative capabilities but raise ethical and security concerns by memorizing sensitive data, reinforcing biases, and producing harmful content. These risks have spurred interest in LLM unlearning, the task of removing knowledge associated with undesirable data from pre-trained models. However, most existing methods assume access to clean, well-defined forget data samples, whereas real-world forget data could often be low-quality, synthetically rewritten, or watermarked, casting doubt on the reliability of unlearning. This work presents the first study of unlearning under perturbed or low-fidelity forget data, referred to as noisy forget sets. By systematically benchmarking state-of-the-art LLM unlearning methods, RMU and NPO, on such noisy forget sets, we find that unlearning remains surprisingly robust to perturbations, provided that core semantic signals are preserved. To explain this robustness, we propose a saliency-based interpretation: key semantic components that drive forgetting remain consistently influential despite substantial variation in surface form. This suggests that unlearning algorithms are primarily guided by deep semantic cues rather than shallow lexical patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01644v2">Machine Learning for Detection and Analysis of Novel LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20711v2">Automating eHMI Action Design with LLMs for Automated Vehicle Communication</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted as findings for EMNLP 2025
    </div>
    <details class="paper-abstract">
      The absence of explicit communication channels between automated vehicles (AVs) and other road users requires the use of external Human-Machine Interfaces (eHMIs) to convey messages effectively in uncertain scenarios. Currently, most eHMI studies employ predefined text messages and manually designed actions to perform these messages, which limits the real-world deployment of eHMIs, where adaptability in dynamic scenarios is essential. Given the generalizability and versatility of large language models (LLMs), they could potentially serve as automated action designers for the message-action design task. To validate this idea, we make three contributions: (1) We propose a pipeline that integrates LLMs and 3D renderers, using LLMs as action designers to generate executable actions for controlling eHMIs and rendering action clips. (2) We collect a user-rated Action-Design Scoring dataset comprising a total of 320 action sequences for eight intended messages and four representative eHMI modalities. The dataset validates that LLMs can translate intended messages into actions close to a human level, particularly for reasoning-enabled LLMs. (3) We introduce two automated raters, Action Reference Score (ARS) and Vision-Language Models (VLMs), to benchmark 18 LLMs, finding that the VLM aligns with human preferences yet varies across eHMI modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03243v2">Prompt-Aware Scheduling for Low-Latency LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Efficient scheduling of LLM inference tasks is essential for achieving low latency and high throughput, particularly with the growing use of reasoning-capable LLMs. Traditional strategies like First-Come-First-Serve (FCFS) often suffer from Head-of-Line (HOL) blocking, where long-running tasks delay shorter ones queued behind them. In this paper, we introduce PARS, a prompt-aware LLM task scheduler that improves serving efficiency by approximating shortest-job-first (SJF) scheduling through pairwise ranking with margin ranking loss. PARS focuses on impactful scheduling decisions and is seamlessly integrated into the state-of-the-art LLM serving system vLLM. It effectively predicts response-length-based task ordering, reducing latency with minimal overhead. Extensive experiments across multiple LLMs and real-world inference datasets show that PARS significantly improves performance, including for reasoning workloads. Furthermore, our cross-model evaluations demonstrate that the design generalizes well, enabling effective scheduling even when predictors are trained on different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08988v1">MASA: LLM-Driven Multi-Agent Systems for Autoformalization</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 EMNLP 2025 Demo camera-ready. Code and data are available at: https://github.com/lanzhang128/multi_agent_autoformalization
    </div>
    <details class="paper-abstract">
      Autoformalization serves a crucial role in connecting natural language and formal reasoning. This paper presents MASA, a novel framework for building multi-agent systems for autoformalization driven by Large Language Models (LLMs). MASA leverages collaborative agents to convert natural language statements into their formal representations. The architecture of MASA is designed with a strong emphasis on modularity, flexibility, and extensibility, allowing seamless integration of new agents and tools to adapt to a fast-evolving field. We showcase the effectiveness of MASA through use cases on real-world mathematical definitions and experiments on formal mathematics datasets. This work highlights the potential of multi-agent systems powered by the interaction of LLMs and theorem provers in enhancing the efficiency and reliability of autoformalization, providing valuable insights and support for researchers and practitioners in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09014v2">Learning to Reason Across Parallel Samples for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Scaling test-time compute brings substantial performance gains for large language models (LLMs). By sampling multiple answers and heuristically aggregate their answers (e.g., either through majority voting or using verifiers to rank the answers), one can achieve consistent performance gains in math domains. In this paper, we propose a new way to leverage such multiple sample set. We train a compact LLM, called Sample Set Aggregator (SSA), that takes a concatenated sequence of multiple samples and output the final answer, optimizing it for the answer accuracy with reinforcement learning. Experiments on five reasoning datasets demonstrate both the efficacy and efficiency of SSA. Notably, SSA improves over naive majority voting by 8% pass@5 on MATH. Furthermore, our 3B SSA surpasses model-based re-ranking with a much larger 72B process reward model. Our analysis also shows promising generalization ability of SSA, across sample set sizes, base model families and scales, and tasks. By separating LLMs to generate answers and LLMs to analyze and aggregate sampled answers, our approach can work with the outputs from premier black box models easily and efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07733v2">SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08952v1">When LLM Agents Meet Graph Optimization: An Automated Data Quality Improvement Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages, 7figures
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) have emerged as a powerful representation that combines structural connections with fine-grained semantics, supporting a wide range of data-centric applications. However, the performance of graph neural networks (GNNs) on TAGs is highly sensitive to input quality. Our empirical study shows that both traditional GNNs and LLM-enhanced GNNs suffer significant degradation across nine representative scenarios of sparsity, noise, and imbalance, highlighting graph quality as a critical bottleneck. Existing approaches mainly focus on improving model architectures, while neglecting systematic optimization of TAG data itself, leading to limited effectiveness in practice. To address this gap, we propose LAGA (Large Language and Graph Agent), a unified multi-agent framework that treats graph quality control as a first-class, data-centric problem. LAGA integrates four collaborative agents-detection, planning, action, and evaluation-into an automated closed loop. At its core, the action agent employs a dual-encoder and tri-objective design to capture complementary information across modalities and perform holistic graph quality enhancement. Experiments across nine scenarios show that LAGA improves graph quality and achieves state-of-the-art performance across various tasks and backbones, validating data-centric quality optimization as key to reliable TAGs and robust graph learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08948v1">SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from JD.com demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on JD.com has substantially enhanced the efficiency of case investigation workflows for risk managers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01605v2">Medchain: Bridging the Gap Between LLM Agents and Clinical Practice with Interactive Sequence</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by NeurIPS25 Spotlight
    </div>
    <details class="paper-abstract">
      Clinical decision making (CDM) is a complex, dynamic process crucial to healthcare delivery, yet it remains a significant challenge for artificial intelligence systems. While Large Language Model (LLM)-based agents have been tested on general medical knowledge using licensing exams and knowledge question-answering tasks, their performance in the CDM in real-world scenarios is limited due to the lack of comprehensive testing datasets that mirror actual medical practice. To address this gap, we present MedChain, a dataset of 12,163 clinical cases that covers five key stages of clinical workflow. MedChain distinguishes itself from existing benchmarks with three key features of real-world clinical practice: personalization, interactivity, and sequentiality. Further, to tackle real-world CDM challenges, we also propose MedChain-Agent, an AI system that integrates a feedback mechanism and a MCase-RAG module to learn from previous cases and adapt its responses. MedChain-Agent demonstrates remarkable adaptability in gathering information dynamically and handling sequential clinical tasks, significantly outperforming existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08931v1">RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Data contamination poses a significant challenge to reliable LLM evaluation, where models may achieve high performance by memorizing training data rather than demonstrating genuine reasoning capabilities. We introduce RADAR (Recall vs. Reasoning Detection through Activation Representation), a novel framework that leverages mechanistic interpretability to detect contamination by distinguishing recall-based from reasoning-based model responses. RADAR extracts 37 features spanning surface-level confidence trajectories and deep mechanistic properties including attention specialization, circuit dynamics, and activation flow patterns. Using an ensemble of classifiers trained on these features, RADAR achieves 93\% accuracy on a diverse evaluation set, with perfect performance on clear cases and 76.7\% accuracy on challenging ambiguous examples. This work demonstrates the potential of mechanistic interpretability for advancing LLM evaluation beyond traditional surface-level metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18630v2">DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong generalization and reasoning abilities, making them well-suited for complex decision-making tasks such as medical consultation (MC). However, existing LLM-based methods often fail to capture the dual nature of MC, which entails two distinct sub-tasks: symptom inquiry, a sequential decision-making process, and disease diagnosis, a classification problem. This mismatch often results in ineffective symptom inquiry and unreliable disease diagnosis. To address this, we propose \textbf{DDO}, a novel LLM-based framework that performs \textbf{D}ual-\textbf{D}ecision \textbf{O}ptimization by decoupling the two sub-tasks and optimizing them with distinct objectives through a collaborative multi-agent workflow. Experiments on three real-world MC datasets show that DDO consistently outperforms existing LLM-based approaches and achieves competitive performance with state-of-the-art generation-based methods, demonstrating its effectiveness in the MC task. The code is available at https://github.com/zh-jia/DDO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04996v2">Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16548v2">Exploring human-SAV interaction using LLMs: The impact of psychological factors on user experience</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      There has been extensive prior work exploring how psychological factors such as anthropomorphism affect the adoption of Shared Autonomous Vehicles (SAVs). However, limited research has been conducted on how prompt strategies in large language models (LLM)-powered conversational SAV agents affect users' perceptions, experiences, and intentions to adopt such technology. In this work, we investigate how conversational SAV agents powered by LLMs drive these psychological factors, such as psychological ownership, the sense of possession a user may come to feel towards an entity or object they may not legally own. We designed four SAV agents with varying levels of anthropomorphic characteristics and psychological ownership triggers. Quantitative measures of psychological ownership, anthropomorphism, quality of service, disclosure tendency, sentiment of SAV responses, and overall acceptance were collected after participants interacted with each SAV. Qualitative feedback was also gathered regarding the experience of psychological ownership during the interactions. The results indicate that an SAV designed to be more anthropomorphic and to induce psychological ownership improved users' perceptions of the SAV's human-like qualities, and its responses were perceived as more positive but also more subjective compared to the control conditions. Qualitative findings support established routes to psychological ownership in the SAV context and suggest that the conversational agent's perceived performance may also influence psychological ownership. Both quantitative and qualitative outcomes highlight the importance of personalization in designing effective SAV interactions. These findings provide practical guidance for designing conversational SAV agents that enhance user experience and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07458v2">Populism Meets AI: Advancing Populism Research with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 27 pages, 3 figures. Preprint version under review
    </div>
    <details class="paper-abstract">
      Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02261v2">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Sequential recommendation systems aspire to profile users by interpreting their interaction histories, echoing how humans make decisions by weighing experience, relative preference strength, and situational relevance. Yet, existing large language model (LLM)-based recommenders often fall short of mimicking the flexible, context-aware decision strategies humans exhibit, neglecting the structured, dynamic, and context-aware mechanisms fundamental to human behaviors. To bridge this gap, we propose RecPO, a preference optimization framework that models structured feedback and contextual delay to emulate human-like prioritization in sequential recommendation. RecPO exploits adaptive reward margins based on inferred preference hierarchies and temporal signals, enabling the model to favor immediately relevant items and to distinguish between varying degrees of preference and aversion. Extensive experiments across five real-world datasets demonstrate that RecPO not only yields performance gains over state-of-the-art baselines, but also mirrors key characteristics of human decision-making: favoring timely satisfaction, maintaining coherent preferences, and exercising discernment under shifting contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v1">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04503v2">P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08886v1">FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The complexity of the Generally Accepted Accounting Principles (GAAP) and the hierarchical structure of eXtensible Business Reporting Language (XBRL) filings make financial auditing increasingly difficult to automate and verify. While large language models (LLMs) have demonstrated strong capabilities in unstructured text understanding, their ability to reason over structured, interdependent, and taxonomy-driven financial documents remains largely unexplored. To fill this gap, we introduce FinAuditing, the first taxonomy-aligned, structure-aware, multi-document benchmark for evaluating LLMs on financial auditing tasks. Built from real US-GAAP-compliant XBRL filings, FinAuditing defines three complementary subtasks, FinSM for semantic consistency, FinRE for relational consistency, and FinMR for numerical consistency, each targeting a distinct aspect of structured auditing reasoning. We further propose a unified evaluation framework integrating retrieval, classification, and reasoning metrics across these subtasks. Extensive zero-shot experiments on 13 state-of-the-art LLMs reveal that current models perform inconsistently across semantic, relational, and mathematical dimensions, with accuracy drops of up to 60-90% when reasoning over hierarchical multi-document structures. Our findings expose the systematic limitations of modern LLMs in taxonomy-grounded financial reasoning and establish FinAuditing as a foundation for developing trustworthy, structure-aware, and regulation-aligned financial intelligence systems. The benchmark dataset is available at Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20650v2">FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Accurately understanding numbers from financial reports is fundamental to how markets, regulators, algorithms, and normal people read the economy and the world, yet even with XBRL (eXtensible Business Reporting Language) designed to tag every figure with standardized accounting concepts, mapping thousands of facts to over 10,000 U.S. GAAP concepts remains costly, inconsistent, and error-prone. Existing benchmarks define tagging as flat, single-step, extreme classification over small subsets of US-GAAP concepts, overlooking both the taxonomy's hierarchical semantics and the structured nature of real tagging, where each fact must be represented as a contextualized multi-field output. These simplifications prevent fair evaluation of large language models (LLMs) under realistic reporting conditions. To address these gaps, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging, designed to evaluate LLMs' ability to extract and align financial facts through numerical reasoning and taxonomy alignment across text and tables. We define two subtasks: FinNI for numeric identification, which extracts numerical entities and their types from XBRL reports, and FinCL for concept linking, which maps each extracted entity to the corresponding concept in the full US-GAAP taxonomy. Together, these subtasks produce a structured representation of each financial fact. We evaluate diverse LLMs under zero-shot settings and analyze their performance across both subtasks and overall tagging accuracy. Results show that LLMs generalize well in numeric identification but struggle with fine-grained concept linking, revealing current limitations in structure-aware reasoning for accurate financial disclosure. All code and datasets are available on GitHub and Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08872v1">GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at https://github.com/ulab-uiuc/GTAlign .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00350v2">OrcaLoca: An LLM Agent Framework for Software Issue Localization</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08572v1">BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v3">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating the underlying large proof search spaces. While the existing approaches primarily rely on value functions and/or Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Tree Search (BFS) remains underexplored. In this paper, we investigate whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present BFS-Prover, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. BFS-Prover achieves a state-of-the-art score of $72.95\%$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. To facilitate further research and development in this area, we have open-sourced our model at https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08544v1">SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in recent years, driving up the demand for inference. LLM inference is composed of two phases with distinct characteristics: a compute-bound prefill phase followed by a memory-bound decode phase. To efficiently serve LLMs, prior work proposes prefill-decode disaggregation to run each phase on separate hardware. However, existing hardware poorly matches the different requirements of each phase. Current datacenter GPUs and TPUs follow a more-is-better design philosophy that maximizes compute and memory resources, causing memory bandwidth underutilization in the prefill phase and compute underutilization in the decode phase. Such underutilization directly translates into increased serving costs. This paper proposes SPAD (Specialized Prefill and Decode hardware), adopting a less-is-more methodology to design specialized chips tailored to the distinct characteristics of prefill and decode phases. The proposed Prefill Chips have larger systolic arrays and use cost-effective GDDR memory, whereas the proposed Decode Chips retain high memory bandwidth but reduce compute capacity. Compared to modeled H100s, simulations show that the proposed Prefill Chips deliver 8% higher prefill performance on average at 52% lower hardware cost, while the proposed Decode Chips achieve 97% of the decode performance with 28% lower TDP. End-to-end simulations on production traces show that SPAD reduces hardware cost by 19%-41% and TDP by 2%-17% compared to modeled baseline clusters while offering the same performance. Even when models and workloads change, SPAD can reallocate either type of chip to run either phase and still achieve 11%-43% lower hardware costs, demonstrating the longevity of the SPAD design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08517v1">CaRT: Teaching LLM Agents to Know When They Know Enough</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Many tasks require learned models to strategically gather relevant information over multiple rounds of interaction before actually acting on a task. Strategic information gathering requires models to know not only how to effectively acquire information, but also when to stop gathering information and make a decision, in order to avoid overthinking or getting derailed when acting. In this paper, we formalize this problem and introduce Counterfactuals and Reasoning for Termination (CaRT), an approach for teaching LLMs when to stop seeking information. To appropriately learn when to terminate, CaRT fine-tunes LLMs using counterfactual pairs of trajectories, one where termination is appropriate and a minimally modified version of the same trajectory where it is not. It trains the LLM to explain the rationale for the termination decision in either case via verbal reasoning, and imbues this capability into the base LLM via fine-tuning. We instantiate CaRT in two domains: interactive medical diagnosis and math problem solving. In both domains, we find that CaRT improves the efficiency of information gathering and task success rate compared to other fine-tuning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06493v2">Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v4">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems. Our code is available at https://github.com/karannb/interact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18085v2">Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11194v3">Evaluating LLMs' Mathematical Reasoning in Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 26 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), excel in natural language understanding, but their capability for complex mathematical reasoning with an amalgamation of structured tables and unstructured text is uncertain. This study explores LLMs' mathematical reasoning on four financial tabular question-answering datasets: TATQA, FinQA, ConvFinQA, and Multihiertt. Through extensive experiments with various models and prompting techniques, we assess how LLMs adapt to complex tables and mathematical tasks. We focus on sensitivity to table complexity and performance variations with an increasing number of arithmetic reasoning steps. The results provide insights into LLMs' capabilities and limitations in handling complex mathematical scenarios for semi-structured tables. Ultimately, we introduce a novel prompting technique tailored to semi-structured documents, matching or outperforming other baselines in performance while providing a nuanced understanding of LLMs abilities for such a task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08481v1">Forecasting the Buzz: Enriching Hashtag Popularity Prediction with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to CIKM 2025
    </div>
    <details class="paper-abstract">
      Hashtag trends ignite campaigns, shift public opinion, and steer millions of dollars in advertising spend, yet forecasting which tag goes viral is elusive. Classical regressors digest surface features but ignore context, while large language models (LLMs) excel at contextual reasoning but misestimate numbers. We present BuzzProphet, a reasoning-augmented hashtag popularity prediction framework that (1) instructs an LLM to articulate a hashtag's topical virality, audience reach, and timing advantage; (2) utilizes these popularity-oriented rationales to enrich the input features; and (3) regresses on these inputs. To facilitate evaluation, we release HashView, a 7,532-hashtag benchmark curated from social media. Across diverse regressor-LLM combinations, BuzzProphet reduces RMSE by up to 2.8% and boosts correlation by 30% over baselines, while producing human-readable rationales. Results demonstrate that using LLMs as context reasoners rather than numeric predictors injects domain insight into tabular models, yielding an interpretable and deployable solution for social media trend forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14756v2">LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration-exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP)). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing. The code to reproduce the results can be found at https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08439v1">xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 24 Pages, 4 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Modern LLM deployments confront a widening cost-performance spectrum: premium models deliver strong reasoning but are expensive, while lightweight models are economical yet brittle on complex tasks. Static escalation rules and keyword heuristics under-utilize this spectrum and fail to adapt across task types. We present xRouter, a tool-calling-based routing system in which a learned router can either answer directly or invoke one or more external models. The router is trained end-to-end with reinforcement learning using an explicit, cost-aware reward that encodes cost-performance trade-offs, eliminating the need for hand-engineered routing rules. Our implementation encompasses the full reinforcement learning framework, including reward and cost accounting, as well as the deployment and evaluation pipelines. Across diverse benchmarks, xRouter achieves strong cost-performance trade-offs (e.g., substantial cost reductions at comparable task completion rates), and provides empirical insights into what reliably helps learned routing and what does not, ranging from model trainability to the difficulty of eliciting sophisticated orchestration behaviors in small open models. We hope these findings and our open implementation will serve as a practical substrate for advancing learned, cost-aware LLM orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07231v2">Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17320v2">AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Understanding the internal representations of large language models (LLMs) remains a central challenge for interpretability research. Sparse autoencoders (SAEs) offer a promising solution by decomposing activations into interpretable features, but existing approaches rely on fixed sparsity constraints that fail to account for input complexity. We propose AdaptiveK SAE (Adaptive Top K Sparse Autoencoders), a novel framework that dynamically adjusts sparsity levels based on the semantic complexity of each input. Leveraging linear probes, we demonstrate that context complexity is linearly encoded in LLM representations, and we use this signal to guide feature allocation during training. Experiments across ten language models (from 70M to 14B parameters) demonstrate that this complexity-driven adaptation significantly outperforms fixed-sparsity approaches on reconstruction fidelity, explained variance, cosine similarity and interpretability metrics while eliminating the computational burden of extensive hyperparameter tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v3">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20435v2">The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Existing interpretability methods for Large Language Models (LLMs) often fall short by focusing on linear directions or isolated features, overlooking the high-dimensional, nonlinear, and relational geometry within model representations. This study focuses on how adversarial inputs systematically affect the internal representation spaces of LLMs, a topic which remains poorly understood. We propose persistent homology (PH), a tool from topological data analysis, as a principled framework to characterize the multi-scale dynamics within LLM activations. Using PH, we systematically analyze six state-of-the-art models under two distinct adversarial conditions, indirect prompt injection and backdoor fine-tuning, and identify a consistent topological signature of adversarial influence. Across architectures and model sizes, adversarial inputs induce ``topological compression'', where the latent space becomes structurally simpler, collapsing from varied, compact, small-scale features into fewer, dominant, and more dispersed large-scale ones. This topological signature is statistically robust across layers, highly discriminative, and provides interpretable insights into how adversarial effects emerge and propagate. By quantifying the shape of activations and neuronal information flow, our architecture-agnostic framework reveals fundamental invariants of representational change, offering a complementary perspective to existing interpretability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v1">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20749v2">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16723v2">LLM Fingerprinting via Semantically Conditioned Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Most LLM fingerprinting methods teach the model to respond to a few fixed queries with predefined atypical responses (keys). This memorization often does not survive common deployment steps such as finetuning or quantization, and such keys can be easily detected and filtered from LLM responses, ultimately breaking the fingerprint. To overcome these limitations we introduce LLM fingerprinting via semantically conditioned watermarks, replacing fixed query sets with a broad semantic domain, and replacing brittle atypical keys with a statistical watermarking signal diffused throughout each response. After teaching the model to watermark its responses only to prompts from a predetermined domain e.g., French language, the model owner can use queries from that domain to reliably detect the fingerprint and verify ownership. As we confirm in our thorough experimental evaluation, our fingerprint is both stealthy and robust to all common deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08638v4">Examining Multilingual Embedding Models Cross-Lingually Through LLM-Generated Adversarial Examples</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 To appear in EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      The evaluation of cross-lingual semantic search models is often limited to existing datasets from tasks such as information retrieval and semantic textual similarity. We introduce Cross-Lingual Semantic Discrimination (CLSD), a lightweight evaluation task that requires only parallel sentences and a Large Language Model (LLM) to generate adversarial distractors. CLSD measures an embedding model's ability to rank the true parallel sentence above semantically misleading but lexically similar alternatives. As a case study, we construct CLSD datasets for German--French in the news domain. Our experiments show that models fine-tuned for retrieval tasks benefit from pivoting through English, whereas bitext mining models perform best in direct cross-lingual settings. A fine-grained similarity analysis further reveals that embedding models differ in their sensitivity to linguistic perturbations. We release our code and datasets under AGPL-3.0: https://github.com/impresso/cross_lingual_semantic_discrimination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16567v3">Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19033v2">SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Tsinghua&Xiaohongshu
    </div>
    <details class="paper-abstract">
      Existing retrieval-augmented code generation (RACG) methods typically use an external retrieval module to fetch semantically similar code snippets used for generating subsequent fragments. However, even for consecutive code fragments, the content often diverges due to logical progression, resulting in a content gap. This gap undermines the performance of current RACG methods, as \textit{external} retrieval modules based on content matching fail to infer the specific information need of LLMs to generate the next code fragment. Therefore, we propose \textbf{SelfRACG}, a novel paradigm that enables large language models (LLMs) to \textbf{Self}-express their information needs to enhance \textbf{RACG}. Specifically, SelfRACG includes an information need expression module and a two-stage information need-guided training strategy, which encourages LLMs to express their information need. Extensive experiments demonstrate that SelfRACG can retrieve external knowledge that better aligns with the LLM's own information needs, resulting in superior generation performance compared to vanilla RACG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10063v3">Hallucination Detection in LLMs with Topological Divergence on Attention Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08255v1">Opponent Shaping in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 29 pages, 15 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed as autonomous agents in real-world environments. As these deployments scale, multi-agent interactions become inevitable, making it essential to understand strategic behavior in such systems. A central open question is whether LLM agents, like reinforcement learning agents, can shape the learning dynamics and influence the behavior of others through interaction alone. In this paper, we present the first investigation of opponent shaping (OS) with LLM-based agents. Existing OS algorithms cannot be directly applied to LLMs, as they require higher-order derivatives, face scalability constraints, or depend on architectural components that are absent in transformers. To address this gap, we introduce ShapeLLM, an adaptation of model-free OS methods tailored for transformer-based agents. Using ShapeLLM, we examine whether LLM agents can influence co-players' learning dynamics across diverse game-theoretic environments. We demonstrate that LLM agents can successfully guide opponents toward exploitable equilibria in competitive games (Iterated Prisoner's Dilemma, Matching Pennies, and Chicken) and promote coordination and improve collective welfare in cooperative games (Iterated Stag Hunt and a cooperative version of the Prisoner's Dilemma). Our findings show that LLM agents can both shape and be shaped through interaction, establishing opponent shaping as a key dimension of multi-agent LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08242v1">Simulating Teams with LLM Agents: Interactive 2D Environments for Studying Human-AI Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Enabling users to create their own simulations offers a powerful way to study team dynamics and performance. We introduce VirTLab, a system that allows researchers and practitioners to design interactive, customizable simulations of team dynamics with LLM-based agents situated in 2D spatial environments. Unlike prior frameworks that restrict scenarios to predefined or static tasks, our approach enables users to build scenarios, assign roles, and observe how agents coordinate, move, and adapt over time. By bridging team cognition behaviors with scalable agent-based modeling, our system provides a testbed for investigating how environments influence coordination, collaboration, and emergent team behaviors. We demonstrate its utility by aligning simulated outcomes with empirical evaluations and a user study, underscoring the importance of customizable environments for advancing research on multi-agent simulations. This work contributes to making simulations accessible to both technical and non-technical users, supporting the design, execution, and analysis of complex multi-agent experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08233v1">Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) are promising alternatives to autoregressive large language models (AR-LLMs), as they potentially allow higher inference throughput. Reinforcement learning (RL) is a crucial component for dLLMs to achieve comparable performance with AR-LLMs on important tasks, such as reasoning. However, RL algorithms that are well-suited for dLLMs' unique characteristics have yet to be developed. This paper proposes Distribution Matching Policy Optimization (DMPO), a principled and theoretically grounded RL fine-tuning method specifically designed to enhance the reasoning capabilities of dLLMs by matching the dLLM policy distribution to the optimal, reward-tilted one through cross-entropy optimization. We identify a key challenge in the implementation with a small training batch size and propose several effective solutions through a novel weight baseline subtraction technique. DMPO exhibits superior performance on multiple reasoning benchmarks without supervised fine-tuning, with an accuracy improvement of up to $42.9\%$ over previously SOTA baselines and $55.8\%$ over the base model, underscoring the effectiveness of the distribution matching framework. Our code is available at https://github.com/yuchen-zhu-zyc/DMPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08211v1">LLMs Learn to Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Previous research has shown that LLMs finetuned on malicious or incorrect completions within narrow domains (e.g., insecure code or incorrect medical advice) can become broadly misaligned to exhibit harmful behaviors, which is called emergent misalignment. In this work, we investigate whether this phenomenon can extend beyond safety behaviors to a broader spectrum of dishonesty and deception under high-stakes scenarios (e.g., lying under pressure and deceptive behavior). To explore this, we finetune open-sourced LLMs on misaligned completions across diverse domains. Experimental results demonstrate that LLMs show broadly misaligned behavior in dishonesty. Additionally, we further explore this phenomenon in a downstream combined finetuning setting, and find that introducing as little as 1% of misalignment data into a standard downstream task is sufficient to decrease honest behavior over 20%. Furthermore, we consider a more practical human-AI interaction environment where we simulate both benign and biased users to interact with the assistant LLM. Notably, we find that the assistant can be misaligned unintentionally to exacerbate its dishonesty with only 10% biased user population. In summary, we extend the study of emergent misalignment to the domain of dishonesty and deception under high-stakes scenarios, and demonstrate that this risk arises not only through direct finetuning, but also in downstream mixture tasks and practical human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08188v1">MetricalARGS: A Taxonomy for Studying Metrical Poetry with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Prior NLP work studying poetry has focused primarily on automatic poem generation and summarization. Many languages have well-studied traditions of poetic meter which enforce constraints on a poem in terms of syllable and phoneme patterns. Such advanced literary forms offer opportunities for probing deeper reasoning and language understanding in Large Language Models (LLMs) and their ability to follow strict pre-requisites and rules. In this paper, we introduce MetricalARGS, the first taxonomy of poetry-related NLP tasks designed to evaluate LLMs on metrical poetry across four dimensions: Analysis, Retrieval, Generation, and Support. We discuss how these tasks relate to existing NLP tasks, addressing questions around datasets and evaluation metrics. Taking Telugu as our example language, we illustrate how the taxonomy can be used in practice. MetricalARGS highlights the broader possibilities for understanding the capabilities and limitations of today's LLMs through the lens of metrical poetry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22745v2">Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have increasingly adopted the Mixture-of-Experts (MoE) architecture for efficiency. MoE-based LLMs heavily depend on a superficial safety mechanism in which harmful inputs are routed safety-critical experts. However, our analysis reveals that routing decisions for harmful inputs drift significantly after fine-tuning, exposing a critical vulnerability to harmful fine-tuning (HFT) attacks. Existing defenses, primarily designed for monolithic LLMs, are less effective for MoE LLMs as they fail to prevent drift in harmful input routing. To address this limitation, we propose SafeMoE, a safe fine-tuning method tailored to MoE LLMs. SafeMoE directly mitigates routing drift by penalizing the gap between the routing weights of a fine-tuned model and those of the initial safety-aligned model, thereby preserving the safety-aligned routing of harmful inputs to safety-critical experts. Experiments on open-source MoE LLMs ranging from 7B to 141B parameters demonstrate that SafeMoE effectively mitigates HFT attacks, reducing the harmfulness score of OLMoE from 62.0 to 5.0, for example, while maintaining task utility within 1% degradation and incurring only 2% overhead. It significantly outperforms state-of-the-art defense methods for safeguarding LLM fine-tuning and remains effective in recent large-scale MoE LLMs such as gpt-oss and Llama 4. Our implementation is available at https://anonymous.4open.science/r/SafeMoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11112v2">Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06223v2">A Multimodal GUI Architecture for Interfacing with LLM-Based Conversational Assistants</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 24 pages, 19 figures, code available at https://github.com/hansvdam/langbar
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) and real-time speech recognition now make it possible to issue any graphical user interface (GUI) action through natural language and receive the corresponding system response directly through the GUI. Most production applications were never designed with speech in mind. This article provides a concrete architecture that enables GUIs to interface with LLM-based speech-enabled assistants. The architecture makes an application's navigation graph and semantics available through the Model Context Protocol (MCP). The ViewModel, part of the MVVM (Model-View-ViewModel) pattern, exposes the application's capabilities to the assistant by supplying both tools applicable to a currently visible view and application-global tools extracted from the GUI tree router. This architecture facilitates full voice accessibility while ensuring reliable alignment between spoken input and the visual interface, accompanied by consistent feedback across modalities. It future-proofs apps for upcoming OS super assistants that employ computer use agents (CUAs) and natively consume MCP if an application provides it. To address concerns about privacy and data security, the practical effectiveness of locally deployable, open-weight LLMs for speech-enabled multimodal UIs is evaluated. Findings suggest that recent smaller open-weight models approach the performance of leading proprietary models in overall accuracy and require enterprise-grade hardware for fast responsiveness. A demo implementation of the proposed architecture can be found at https://github.com/hansvdam/langbar
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08158v1">Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04810v2">Dissecting Logical Reasoning in LLMs: A Fine-Grained Evaluation and Supervision Study</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by the Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Logical reasoning is a core capability for large language models (LLMs), yet existing benchmarks that rely solely on final-answer accuracy fail to capture the quality of the reasoning process. To address this, we introduce FineLogic, a fine-grained evaluation framework that assesses logical reasoning across three dimensions: overall accuracy, stepwise soundness, and representation-level probing. Leveraging this framework, we conduct a comprehensive study on how different supervision formats in fine-tuning shape reasoning abilities. We fine-tune LLMs on four supervision styles: one in natural language and three symbolic variants. We find a key trade-off: natural language supervision excels at generalization to out-of-distribution and long-chain problems, whereas symbolic supervision is superior at instilling structurally sound, atomic reasoning steps. Furthermore, our probing analysis indicates that fine-tuning primarily refines the model's step-by-step generation process, rather than improving its ability to converge on an answer early. Together, our framework and analysis provide a more rigorous lens for evaluating and improving logical reasoning in LLMs. The code is available at https://github.com/YujunZhou/FineLogic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08120v1">Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 12 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Using LLMs to evaluate text, that is, LLM-as-a-judge, is increasingly being used at scale to augment or even replace human annotations. As such, it is imperative that we understand the potential biases and risks of doing so. In this work, we propose an approach for extracting high-level concept-based global policies from LLM-as-a-Judge. Our approach consists of two algorithms: 1) CLoVE (Contrastive Local Verifiable Explanations), which generates verifiable, concept-based, contrastive local explanations and 2) GloVE (Global Verifiable Explanations), which uses iterative clustering, summarization and verification to condense local rules into a global policy. We evaluate GloVE on seven standard benchmarking datasets for content harm detection. We find that the extracted global policies are highly faithful to decisions of the LLM-as-a-Judge. Additionally, we evaluated the robustness of global policies to text perturbations and adversarial attacks. Finally, we conducted a user study to evaluate user understanding and satisfaction with global policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02886v4">TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by EMNLP2025
    </div>
    <details class="paper-abstract">
      Rapid advances in Large Language Models (LLMs) have spurred demand for processing extended context sequences in contemporary applications. However, this progress faces two challenges: performance degradation due to sequence lengths out-of-distribution, and excessively long inference times caused by the quadratic computational complexity of attention. These issues limit LLMs in long-context scenarios. In this paper, we propose Dynamic Token-Level KV Cache Selection (TokenSelect), a training-free method for efficient and accurate long-context inference. TokenSelect builds upon the observation of non-contiguous attention sparsity, using QK dot products to measure per-head KV Cache criticality at token-level. By per-head soft voting mechanism, TokenSelect selectively involves a few critical KV cache tokens in attention calculation without sacrificing accuracy. To further accelerate TokenSelect, we design the Selection Cache based on observations of consecutive Query similarity and implemented the efficient Paged Dot Product Kernel, significantly reducing the selection overhead. A comprehensive evaluation of TokenSelect demonstrates up to $23.84\times$ speedup in attention computation and up to $2.28\times$ acceleration in end-to-end latency, while providing superior performance compared to state-of-the-art long-context inference methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08111v1">Evaluating LLM-Generated Legal Explanations for Regulatory Compliance in Social Media Influencer Marketing</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted for publication at the Natural Legal Language Processing Workshop (NLLP) 2025, co-located with EMNLP
    </div>
    <details class="paper-abstract">
      The rise of influencer marketing has blurred boundaries between organic content and sponsored content, making the enforcement of legal rules relating to transparency challenging. Effective regulation requires applying legal knowledge with a clear purpose and reason, yet current detection methods of undisclosed sponsored content generally lack legal grounding or operate as opaque "black boxes". Using 1,143 Instagram posts, we compare gpt-5-nano and gemini-2.5-flash-lite under three prompting strategies with controlled levels of legal knowledge provided. Both models perform strongly in classifying content as sponsored or not (F1 up to 0.93), though performance drops by over 10 points on ambiguous cases. We further develop a taxonomy of reasoning errors, showing frequent citation omissions (28.57%), unclear references (20.71%), and hidden ads exhibiting the highest miscue rate (28.57%). While adding regulatory text to the prompt improves explanation quality, it does not consistently improve detection accuracy. The contribution of this paper is threefold. First, it makes a novel addition to regulatory compliance technology by providing a taxonomy of common errors in LLM-generated legal reasoning to evaluate whether automated moderation is not only accurate but also legally robust, thereby advancing the transparent detection of influencer marketing content. Second, it features an original dataset of LLM explanations annotated by two students who were trained in influencer marketing law. Third, it combines quantitative and qualitative evaluation strategies for LLM explanations and critically reflects on how these findings can support advertising regulatory bodies in automating moderation processes on a solid legal foundation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08101v1">LLM-Assisted Web Measurements</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 12 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Web measurements are a well-established methodology for assessing the security and privacy landscape of the Internet. However, existing top lists of popular websites commonly used as measurement targets are unlabeled and lack semantic information about the nature of the sites they include. This limitation makes targeted measurements challenging, as researchers often need to rely on ad-hoc techniques to bias their datasets toward specific categories of interest. In this paper, we investigate the use of Large Language Models (LLMs) as a means to enable targeted web measurement studies through their semantic understanding capabilities. Building on prior literature, we identify key website classification tasks relevant to web measurements and construct datasets to systematically evaluate the performance of different LLMs on these tasks. Our results demonstrate that LLMs may achieve strong performance across multiple classification scenarios. We then conduct LLM-assisted web measurement studies inspired by prior work and rigorously assess the validity of the resulting research inferences. Our results demonstrate that LLMs can serve as a practical tool for analyzing security and privacy trends on the Web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08091v1">Everything is Plausible: Investigating the Impact of LLM Rationales on Human Notions of Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      We investigate the degree to which human plausibility judgments of multiple-choice commonsense benchmark answers are subject to influence by (im)plausibility arguments for or against an answer, in particular, using rationales generated by LLMs. We collect 3,000 plausibility judgments from humans and another 13,600 judgments from LLMs. Overall, we observe increases and decreases in mean human plausibility ratings in the presence of LLM-generated PRO and CON rationales, respectively, suggesting that, on the whole, human judges find these rationales convincing. Experiments with LLMs reveal similar patterns of influence. Our findings demonstrate a novel use of LLMs for studying aspects of human cognition, while also raising practical concerns that, even in domains where humans are ``experts'' (i.e., common sense), LLMs have the potential to exert considerable influence on people's beliefs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08081v1">AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Ranking online reviews by their intrinsic quality is a critical task for e-commerce platforms and information services, impacting user experience and business outcomes. However, quality is a domain-dependent and dynamic concept, making its assessment a formidable challenge. Traditional methods relying on hand-crafted features are unscalable across domains and fail to adapt to evolving content patterns, while modern deep learning approaches often produce black-box models that lack interpretability and may prioritize semantics over quality. To address these challenges, we propose AutoQual, an LLM-based agent framework that automates the discovery of interpretable features. While demonstrated on review quality assessment, AutoQual is designed as a general framework for transforming tacit knowledge embedded in data into explicit, computable features. It mimics a human research process, iteratively generating feature hypotheses through reflection, operationalizing them via autonomous tool implementation, and accumulating experience in a persistent memory. We deploy our method on a large-scale online platform with a billion-level user base. Large-scale A/B testing confirms its effectiveness, increasing average reviews viewed per user by 0.79% and the conversion rate of review readers by 0.27%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06780v2">Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08055v1">From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 13 pages, 5 figure, 8 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference in production must meet stringent service-level objectives for both time-to-first-token (TTFT) and time-between-token (TBT) while maximizing throughput under fixed compute, memory, and interconnect budgets. Modern serving systems adopt stall-free scheduling techniques such as chunked prefill, which splits long prompt processing along the token dimension and interleaves prefill with ongoing decode iterations. While effective at stabilizing TBT, chunked prefill incurs substantial overhead in Mixture-of-Experts (MoE) models: redundant expert weight loads increase memory traffic by up to 39% and inflate energy consumption. We propose layered prefill, a new scheduling paradigm that treats transformer layer groups as the primary scheduling unit. By vertically partitioning the model into contiguous layer groups and interleaving prefill and decode across the groups, layered prefill sustains stall-free decoding while eliminating chunk-induced MoE weight reloads. It reduces off-chip bandwidth demand, lowering TTFT by up to 70%, End-to-End latency by 41% and per-token energy by up to 22%. Evaluations show that layered prefill consistently improves the TTFT--TBT Pareto frontier over chunked prefill, reducing expert-load traffic and energy cost while maintaining stall-free decoding. Overall, shifting the scheduling axis from tokens to layers unlocks a new operating regime for high-efficiency, energy-aware LLM serving in co-located environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08044v1">Towards Reliable LLM-based Robot Planning via Combined Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate advanced reasoning abilities, enabling robots to understand natural language instructions and generate high-level plans with appropriate grounding. However, LLM hallucinations present a significant challenge, often leading to overconfident yet potentially misaligned or unsafe plans. While researchers have explored uncertainty estimation to improve the reliability of LLM-based planning, existing studies have not sufficiently differentiated between epistemic and intrinsic uncertainty, limiting the effectiveness of uncertainty estimation. In this paper, we present Combined Uncertainty estimation for Reliable Embodied planning (CURE), which decomposes the uncertainty into epistemic and intrinsic uncertainty, each estimated separately. Furthermore, epistemic uncertainty is subdivided into task clarity and task familiarity for more accurate evaluation. The overall uncertainty assessments are obtained using random network distillation and multi-layer perceptron regression heads driven by LLM features. We validated our approach in two distinct experimental settings: kitchen manipulation and tabletop rearrangement experiments. The results show that, compared to existing methods, our approach yields uncertainty estimates that are more closely aligned with the actual execution outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05582v2">(Token-Level) InfoRMIA: Stronger Membership Inference and Memorization Assessment for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Machine learning models are known to leak sensitive information, as they inevitably memorize (parts of) their training data. More alarmingly, large language models (LLMs) are now trained on nearly all available data, which amplifies the magnitude of information leakage and raises serious privacy risks. Hence, it is more crucial than ever to quantify privacy risk before the release of LLMs. The standard method to quantify privacy is via membership inference attacks, where the state-of-the-art approach is the Robust Membership Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled information-theoretic formulation of membership inference. Our method consistently outperforms RMIA across benchmarks while also offering improved computational efficiency. In the second part of the paper, we identify the limitations of treating sequence-level membership inference as the gold standard for measuring leakage. We propose a new perspective for studying membership and memorization in LLMs: token-level signals and analyses. We show that a simple token-based InfoRMIA can pinpoint which tokens are memorized within generated outputs, thereby localizing leakage from the sequence level down to individual tokens, while achieving stronger sequence-level inference power on LLMs. This new scope rethinks privacy in LLMs and can lead to more targeted mitigation, such as exact unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v4">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19740v4">Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07985v1">Fewer Weights, More Problems: A Practical Attack on LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07968v1">From Defender to Devil? Unintended Risk Interactions Induced by LLM Defenses</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance across various applications, but their deployment in sensitive domains raises significant concerns. To mitigate these risks, numerous defense strategies have been proposed. However, most existing studies assess these defenses in isolation, overlooking their broader impacts across other risk dimensions. In this work, we take the first step in investigating unintended interactions caused by defenses in LLMs, focusing on the complex interplay between safety, fairness, and privacy. Specifically, we propose CrossRiskEval, a comprehensive evaluation framework to assess whether deploying a defense targeting one risk inadvertently affects others. Through extensive empirical studies on 14 defense-deployed LLMs, covering 12 distinct defense strategies, we reveal several alarming side effects: 1) safety defenses may suppress direct responses to sensitive queries related to bias or privacy, yet still amplify indirect privacy leakage or biased outputs; 2) fairness defenses increase the risk of misuse and privacy leakage; 3) privacy defenses often impair safety and exacerbate bias. We further conduct a fine-grained neuron-level analysis to uncover the underlying mechanisms of these phenomena. Our analysis reveals the existence of conflict-entangled neurons in LLMs that exhibit opposing sensitivities across multiple risk dimensions. Further trend consistency analysis at both task and neuron levels confirms that these neurons play a key role in mediating the emergence of unintended behaviors following defense deployment. We call for a paradigm shift in LLM risk evaluation, toward holistic, interaction-aware assessment of defense strategies.
    </details>
</div>
