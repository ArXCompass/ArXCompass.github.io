# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10054v1">Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to ICSME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15902v2">IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinction between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution (OOD) data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v1">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07498v2">Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12185v3">EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13640v2">Qorgau: Evaluating LLM Safety in Kazakh-Russian Bilingual Contexts</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to have the potential to generate harmful content, posing risks to users. While significant progress has been made in developing taxonomies for LLM risks and safety evaluation prompts, most studies have focused on monolingual contexts, primarily in English. However, language- and region-specific risks in bilingual contexts are often overlooked, and core findings can diverge from those in monolingual settings. In this paper, we introduce Qorgau, a novel dataset specifically designed for safety evaluation in Kazakh and Russian, reflecting the unique bilingual context in Kazakhstan, where both Kazakh (a low-resource language) and Russian (a high-resource language) are spoken. Experiments with both multilingual and language-specific LLMs reveal notable differences in safety performance, emphasizing the need for tailored, region-specific datasets to ensure the responsible and safe deployment of LLMs in countries like Kazakhstan. Warning: this paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13820v2">InstCache: A Predictive Cache for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      The revolutionary capabilities of Large Language Models (LLMs) are attracting rapidly growing popularity and leading to soaring user requests to inference serving systems. Caching techniques, which leverage data reuse to reduce computation, offer opportunities to optimize the performance of LLM inference engines. On the one hand, the low-level key-value (KV) cache working at the token level is widely adopted, albeit it incurs significant overhead as request volume grows. On the other hand, instruction-level caching, which stores full instruction-response pairs, is expected to play an increasingly crucial role. However, the high variability in the content and length of instructions make it rare for identical instructions to recur within a short time window, presenting challenges for effective caching instruction-response pairs. To address this challenge, we propose InstCache, a predictive caching mechanism for LLM serving systems. Leveraging the capability of LLMs, we can effectively reorder the representation space of instruction texts and develop a sufficient level of spatial locality. Such spatial locality enables us to predict potential instructions located in a compact region in the space, resulting in an effective caching system at runtime. Experimental results demonstrate that InstCache achieves a 2.3x higher hit rate compared to the upper bound of traditional caching mechanisms on WildChat dataset and reduces the time per output token of vLLM by up to 42.0% and 50.0% on LMSys and Moss datasets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09839v1">Rethinking Prompt Optimization: Reinforcement, Diversification, and Migration in Blackbox LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      An increasing number of NLP applications interact with large language models (LLMs) through black-box APIs, making prompt engineering critical for controlling model outputs. While recent Automatic Prompt Optimization (APO) methods iteratively refine prompts using model-generated feedback, textual gradients, they primarily focus on error correction and neglect valuable insights from correct predictions. This limits both their effectiveness and efficiency. In this paper, we propose a novel APO framework centered on enhancing the feedback mechanism. We reinterpret the textual gradient as a form of negative reinforcement and introduce the complementary positive reinforcement to explicitly preserve beneficial prompt components identified through successful predictions. To mitigate the noise inherent in LLM-generated feedback, we introduce a technique called feedback diversification, which aggregates multiple feedback signals, emphasizing consistent, actionable advice while filtering out outliers. Motivated by the rapid evolution and diversity of available LLMs, we also formalize Continual Prompt Optimization (CPO), addressing the practical challenge of efficiently migrating optimized prompts between different model versions or API providers. Our experiments reveal that naive prompt migration often degrades performance due to loss of critical instructions. In contrast, our approach consistently outperforms strong baselines, achieving significant accuracy improvements, faster convergence, and lower computational costs in both standard and migration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10852v1">LLMs on Trial: Evaluating Judicial Fairness for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in high-stakes fields where their decisions impact rights and equity. However, LLMs' judicial fairness and implications for social justice remain underexplored. When LLMs act as judges, the ability to fairly resolve judicial issues is a prerequisite to ensure their trustworthiness. Based on theories of judicial fairness, we construct a comprehensive framework to measure LLM fairness, leading to a selection of 65 labels and 161 corresponding values. Applying this framework to the judicial system, we compile an extensive dataset, JudiFair, comprising 177,100 unique case facts. To achieve robust statistical inference, we develop three evaluation metrics, inconsistency, bias, and imbalanced inaccuracy, and introduce a method to assess the overall fairness of multiple LLMs across various labels. Through experiments with 16 LLMs, we uncover pervasive inconsistency, bias, and imbalanced inaccuracy across models, underscoring severe LLM judicial unfairness. Particularly, LLMs display notably more pronounced biases on demographic labels, with slightly less bias on substance labels compared to procedure ones. Interestingly, increased inconsistency correlates with reduced biases, but more accurate predictions exacerbate biases. While we find that adjusting the temperature parameter can influence LLM fairness, model size, release date, and country of origin do not exhibit significant effects on judicial fairness. Accordingly, we introduce a publicly available toolkit containing all datasets and code, designed to support future research in evaluating and improving LLM fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10844v1">LLM-Guided Agentic Object Detection for Open-World Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Object detection traditionally relies on fixed category sets, requiring costly re-training to handle novel objects. While Open-World and Open-Vocabulary Object Detection (OWOD and OVOD) improve flexibility, OWOD lacks semantic labels for unknowns, and OVOD depends on user prompts, limiting autonomy. We propose an LLM-guided agentic object detection (LAOD) framework that enables fully label-free, zero-shot detection by prompting a Large Language Model (LLM) to generate scene-specific object names. These are passed to an open-vocabulary detector for localization, allowing the system to adapt its goals dynamically. We introduce two new metrics, Class-Agnostic Average Precision (CAAP) and Semantic Naming Average Precision (SNAP), to separately evaluate localization and naming. Experiments on LVIS, COCO, and COCO-OOD validate our approach, showing strong performance in detecting and naming novel objects. Our method offers enhanced autonomy and adaptability for open-world understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10818v1">How Robust are LLM-Generated Library Imports? An Empirical Study using Stack Overflow</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Software libraries are central to the functionality, security, and maintainability of modern code. As developers increasingly turn to Large Language Models (LLMs) to assist with programming tasks, understanding how these models recommend libraries is essential. In this paper, we conduct an empirical study of six state-of-the-art LLMs, both proprietary and open-source, by prompting them to solve real-world Python problems sourced from Stack Overflow. We analyze the types of libraries they import, the characteristics of those libraries, and the extent to which the recommendations are usable out of the box. Our results show that LLMs predominantly favour third-party libraries over standard ones, and often recommend mature, popular, and permissively licensed dependencies. However, we also identify gaps in usability: 4.6% of the libraries could not be resolved automatically due to structural mismatches between import names and installable packages, and only two models (out of six) provided installation guidance. While the generated code is technically valid, the lack of contextual support places the burden of manually resolving dependencies on the user. Our findings offer actionable insights for both developers and researchers, and highlight opportunities to improve the reliability and usability of LLM-generated code in the context of software dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10803v1">Automated Thematic Analyses Using LLMs: Xylazine Wound Management Social Media Chatter Use Case</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Pages: 19, Abstract word count: 151 words, Manuscript word count: 2185 words, References: 14, Figures: 3, Tables: 2
    </div>
    <details class="paper-abstract">
      Background Large language models (LLMs) face challenges in inductive thematic analysis, a task requiring deep interpretive and domain-specific expertise. We evaluated the feasibility of using LLMs to replicate expert-driven thematic analysis of social media data. Methods Using two temporally non-intersecting Reddit datasets on xylazine (n=286 and n=686, for model optimization and validation, respectively) with twelve expert-derived themes, we evaluated five LLMs against expert coding. We modeled the task as a series of binary classifications, rather than a single, multi-label classification, employing zero-, single-, and few-shot prompting strategies and measuring performance via accuracy, precision, recall, and F1-score. Results On the validation set, GPT-4o with two-shot prompting performed best (accuracy: 90.9%; F1-score: 0.71). For high-prevalence themes, model-derived thematic distributions closely mirrored expert classifications (e.g., xylazine use: 13.6% vs. 17.8%; MOUD use: 16.5% vs. 17.8%). Conclusions Our findings suggest that few-shot LLM-based approaches can automate thematic analyses, offering a scalable supplement for qualitative research. Keywords: thematic analysis, large language models, natural language processing, qualitative analysis, social media, prompt engineering, public health
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04644v2">Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Agentic Reasoning dynamically leverages web search, code execution, and structured memory to address complex problems requiring deep research. A key innovation in our framework is the Mind-Map agent, which constructs a structured knowledge graph to store reasoning context and track logical relationships, ensuring coherence in long reasoning chains with extensive tool usage. Additionally, we conduct a comprehensive exploration of the Web-Search agent, leading to a highly effective search mechanism that surpasses all prior approaches. When deployed on DeepSeek-R1, our method achieves a new state-of-the-art (SOTA) among public models and delivers performance comparable to OpenAI Deep Research, the leading proprietary model in this domain. Extensive ablation studies validate the optimal selection of agentic tools and confirm the effectiveness of our Mind-Map and Web-Search agents in enhancing LLM reasoning. The code is at: https://github.com/theworldofagents/Agentic-Reasoning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10778v1">Warehouse Spatial Question Answering with LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 1st Place Solution of the 9th AI City Challenge Track 3
    </div>
    <details class="paper-abstract">
      Spatial understanding has been a challenging task for existing Multi-modal Large Language Models~(MLLMs). Previous methods leverage large-scale MLLM finetuning to enhance MLLM's spatial understanding ability. In this paper, we present a data-efficient approach. We propose a LLM agent system with strong and advanced spatial reasoning ability, which can be used to solve the challenging spatial question answering task in complex indoor warehouse scenarios. Our system integrates multiple tools that allow the LLM agent to conduct spatial reasoning and API tools interaction to answer the given complicated spatial question. Extensive evaluations on the 2025 AI City Challenge Physical AI Spatial Intelligence Warehouse dataset demonstrate that our system achieves high accuracy and efficiency in tasks such as object retrieval, counting, and distance estimation. The code is available at: https://github.com/hsiangwei0903/SpatialAgent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05209v3">Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02820v4">DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Compound AI systems, such as agentic systems, are an emerging trend in large-scale enterprise settings, with multiple LLMs specialized for different users, tasks, and/or roles working together. In these scenarios, different models often process inputs that share the same context prefix. Although much work was done in the past to enable the reuse of prefix KV caches across inputs for a single model, how to enable one model to reuse the prefix KV caches of a different model remains an open question. We introduce DroidSpeak, the first distributed LLM inference system that enables KV cache reuse across distributed nodes running inference of different LLMs, so long as the LLMs have the same architecture. We present the first study that aims at understanding the impact of sharing KV caches across different LLMs, and if/when such sharing affects quality. Inspired by the findings, we present DroidSpeak, which selectively recomputes a few layers of the KV cache produced by another LLM and reuses the remaining layers, with negligible quality loss. Moreover, carefully pipelining the layer-wise re-computation and the loading of reused KV cache further improves the inference performance. Experiments on diverse datasets and model pairs demonstrate that DroidSpeak achieves up to 4x throughput improvement and about 3.1x faster prefill (time to first token), with negligible loss of quality in F1 scores, Rouge-L or code similarity score, compared to the baseline which does not allow any sharing across models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10695v1">Exploring User Security and Privacy Attitudes and Concerns Toward the Use of General-Purpose LLM Chatbots for Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to the 34th USENIX Security Symposium
    </div>
    <details class="paper-abstract">
      Individuals are increasingly relying on large language model (LLM)-enabled conversational agents for emotional support. While prior research has examined privacy and security issues in chatbots specifically designed for mental health purposes, these chatbots are overwhelmingly "rule-based" offerings that do not leverage generative AI. Little empirical research currently measures users' privacy and security concerns, attitudes, and expectations when using general-purpose LLM-enabled chatbots to manage and improve mental health. Through 21 semi-structured interviews with U.S. participants, we identified critical misconceptions and a general lack of risk awareness. Participants conflated the human-like empathy exhibited by LLMs with human-like accountability and mistakenly believed that their interactions with these chatbots were safeguarded by the same regulations (e.g., HIPAA) as disclosures with a licensed therapist. We introduce the concept of "intangible vulnerability," where emotional or psychological disclosures are undervalued compared to more tangible forms of information (e.g., financial or location-based data). To address this, we propose recommendations to safeguard user mental health disclosures with general-purpose LLM-enabled chatbots more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10639v1">SPICEAssistant: LLM using SPICE Simulation Tools for Schematic Design of Switched-Mode Power Supplies</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art large language models (LLMs) show high performance across a wide range of tasks in many domains of science. In the field of electronic design automation (EDA), it is yet to be determined to what extent they are capable to understand, adapt, and dimension electronic circuits. This paper focuses on the application of LLMs to switched-mode power supply (SMPS) design on printed circuit boards (PCBs). Particular challenges for LLMs in this context include their limited ability to interpret results from key simulation tools like SPICE and the multi-step design process. To address these challenges, we suggest SPICEAssistant, a framework that provides a broad selection of tools to an LLM. The tools serve as an interface to SPICE, allowing the LLM to interact flexibly with the simulator to estimate the impact of its modifications to the circuit. To evaluate the performance of SPICEAssistant, we defined a benchmark consisting of 256 questions testing the ability to adapt circuit netlists to fulfil different SMPS design tasks. The benchmarking results show that simulation feedback effectively improves SMPS design capabilities of LLMs. An increasing number of simulation iterations leads to enhanced performance. The SPICEAssistant framework significantly outperforms the standalone LLM GPT-4o on the benchmark by approximately 38%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10628v1">GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10624v1">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Substantial change to previous version (experiments, theorem, analysis and related work); currently under review at TMLR
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10621v1">Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation. The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset. This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09820v1">Measuring What Matters: A Framework for Evaluating Safety Risks in Real-World LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Most safety testing efforts for large language models (LLMs) today focus on evaluating foundation models. However, there is a growing need to evaluate safety at the application level, as components such as system prompts, retrieval pipelines, and guardrails introduce additional factors that significantly influence the overall safety of LLM applications. In this paper, we introduce a practical framework for evaluating application-level safety in LLM systems, validated through real-world deployment across multiple use cases within our organization. The framework consists of two parts: (1) principles for developing customized safety risk taxonomies, and (2) practices for evaluating safety risks in LLM applications. We illustrate how the proposed framework was applied in our internal pilot, providing a reference point for organizations seeking to scale their safety testing efforts. This work aims to bridge the gap between theoretical concepts in AI safety and the operational realities of safeguarding LLM applications in practice, offering actionable guidance for safe and scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09790v1">Prompting for Performance: Exploring LLMs for Configuring Software</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Software systems usually provide numerous configuration options that can affect performance metrics such as execution time, memory usage, binary size, or bitrate. On the one hand, making informed decisions is challenging and requires domain expertise in options and their combinations. On the other hand, machine learning techniques can search vast configuration spaces, but with a high computational cost, since concrete executions of numerous configurations are required. In this exploratory study, we investigate whether large language models (LLMs) can assist in performance-oriented software configuration through prompts. We evaluate several LLMs on tasks including identifying relevant options, ranking configurations, and recommending performant configurations across various configurable systems, such as compilers, video encoders, and SAT solvers. Our preliminary results reveal both positive abilities and notable limitations: depending on the task and systems, LLMs can well align with expert knowledge, whereas hallucinations or superficial reasoning can emerge in other cases. These findings represent a first step toward systematic evaluations and the design of LLM-based solutions to assist with software configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09788v1">TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 9 pages. Preprint to be submitted to peer-review
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, highlighting possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at https://github.com/microsoft/tinytroupe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v4">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 6 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite gradual improvements in hardware capabilities. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new framework that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose \acronym, a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model. To further increase the efficiency of verification, the edge server batch the diverse verification requests from devices. This approach supports device heterogeneity and reduces server-side memory footprint by sharing the same upstream target model across multiple devices. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: 2.2 more system throughput, 2.8 more system capacity, and better cost efficiency, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11462v5">Cascade Speculative Drafting for Even Faster LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Introduced to enhance the efficiency of large language model (LLM) inference, speculative decoding operates by having a smaller model generate a draft. A larger target model then reviews this draft to align with its output, and any acceptance by the target model results in a reduction of the number of the target model runs, ultimately improving efficiency. However, the drafting process in speculative decoding includes slow autoregressive generation and allocates equal time to generating tokens, irrespective of their importance. These inefficiencies collectively contribute to the suboptimal performance of speculative decoding. To further improve LLM inference, we introduce Cascade Speculative Drafting (CS Drafting), a speculative execution algorithm that incorporates two types of cascades. The Vertical Cascade eliminates autoregressive generation from neural models, while the Horizontal Cascade optimizes time allocation in drafting for improved efficiency. Combining both cascades, CS Drafting achieves greater speedup compared to the baselines in our experiments, while preserving the same output distribution as the target model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09751v1">Sound and Complete Neuro-symbolic Reasoning with LLM-Grounded Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 29 pages, 9 tables, 3 figures. Accepted to the 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neuro-symbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09701v1">MCEval: A Dynamic Framework for Fair Multilingual Cultural Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Large language models exhibit cultural biases and limited cross-cultural understanding capabilities, particularly when serving diverse global user populations. We propose MCEval, a novel multilingual evaluation framework that employs dynamic cultural question construction and enables causal analysis through Counterfactual Rephrasing and Confounder Rephrasing. Our comprehensive evaluation spans 13 cultures and 13 languages, systematically assessing both cultural awareness and cultural bias across different linguistic scenarios. The framework provides 39,897 cultural awareness instances and 17,940 cultural bias instances. Experimental results reveal performance disparities across different linguistic scenarios, demonstrating that optimal cultural performance is not only linked to training data distribution, but also is related to language-culture alignment. The evaluation results also expose the fairness issue, where approaches appearing successful in the English scenario create substantial disadvantages. MCEval represents the first comprehensive multilingual cultural evaluation framework that provides deeper insights into LLMs' cultural understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09657v1">Negotiating Comfort: Simulating Personality-Driven LLM Agents in Shared Residential Social Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      We use generative agents powered by large language models (LLMs) to simulate a social network in a shared residential building, driving the temperature decisions for a central heating system. Agents, divided into Family Members and Representatives, consider personal preferences, personal traits, connections, and weather conditions. Daily simulations involve family-level consensus followed by building-wide decisions among representatives. We tested three personality traits distributions (positive, mixed, and negative) and found that positive traits correlate with higher happiness and stronger friendships. Temperature preferences, assertiveness, and selflessness have a significant impact on happiness and decisions. This work demonstrates how LLM-driven agents can help simulate nuanced human behavior where complex real-life human simulations are difficult to set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08965v2">LLM-Driven Usefulness Labeling for IR Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      In the information retrieval (IR) domain, evaluation plays a crucial role in optimizing search experiences and supporting diverse user intents. In the recent LLM era, research has been conducted to automate document relevance labels, as these labels have traditionally been assigned by crowd-sourced workers - a process that is both time and consuming and costly. This study focuses on LLM-generated usefulness labels, a crucial evaluation metric that considers the user's search intents and task objectives, an aspect where relevance falls short. Our experiment utilizes task-level, query-level, and document-level features along with user search behavior signals, which are essential in defining the usefulness of a document. Our research finds that (i) pre-trained LLMs can generate moderate usefulness labels by understanding the comprehensive search task session, (ii) pre-trained LLMs perform better judgement in short search sessions when provided with search session contexts. Additionally, we investigated whether LLMs can capture the unique divergence between relevance and usefulness, along with conducting an ablation study to identify the most critical metrics for accurate usefulness label generation. In conclusion, this work explores LLM-generated usefulness labels by evaluating critical metrics and optimizing for practicality in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05310v2">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models have proven surprisingly effective at solving a wide range of tasks from just a handful of examples. However, their lack of reliability and modularity limits their capacity to tackle large problems that require many steps of reasoning. In response, researchers have proposed advanced pipelines that leverage domain-specific knowledge to chain smaller prompts, provide intermediate feedback and improve performance through search. However, the current complexity of writing, tuning, maintaining and improving such pipelines has limited their sophistication. We propose oracular programming, a foundational paradigm for building LLM-enabled applications that lets domain experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful search tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09490v1">Towards LLM-Based Automatic Playtest</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Playtesting is the process in which people play a video game for testing. It is critical for the quality assurance of gaming software. Manual playtesting is time-consuming and expensive. However, automating this process is challenging, as playtesting typically requires domain knowledge and problem-solving skills that most conventional testing tools lack. Recent advancements in artificial intelligence (AI) have opened up new possibilities for applying Large Language Models (LLMs) to playtesting. However, significant challenges remain: current LLMs cannot visually perceive game environments, and most existing research focuses on text-based games or games with robust APIs. Many non-text games lack APIs to provide textual descriptions of game states, making it almost impossible to naively apply LLMs for playtesting. This paper introduces Lap, our novel approach to LLM-based Automatic Playtesting, which uses ChatGPT to test match-3 games, a category of games where players match three or more identical tiles in a row or column to earn points. Lap encompasses three key phases: processing of game environments, prompting-based action generation, and action execution. Given a match-3 game, Lap takes a snapshot of the game board and converts it to a numeric matrix. It then prompts the ChatGPT-O1-mini API to suggest moves based on that matrix and tentatively applies the suggested moves to earn points and trigger changes in the game board. It repeats the above-mentioned three steps iteratively until timeout. For evaluation, we conducted a case study using Lap on an open-source match-3 game, CasseBonbons, and empirically compared it with three existing tools. Our results are promising: Lap outperformed existing tools by achieving higher code coverage and triggering more program crashes. This research sheds light on the future of automatic testing and LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09488v1">Criteria-Based LLM Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 10 pages, 3 figures, accepted to ICTIR 2025
    </div>
    <details class="paper-abstract">
      Relevance judgments are crucial for evaluating information retrieval systems, but traditional human-annotated labels are time-consuming and expensive. As a result, many researchers turn to automatic alternatives to accelerate method development. Among these, Large Language Models (LLMs) provide a scalable solution by generating relevance labels directly through prompting. However, prompting an LLM for a relevance label without constraints often results in not only incorrect predictions but also outputs that are difficult for humans to interpret. We propose the Multi-Criteria framework for LLM-based relevance judgments, decomposing the notion of relevance into multiple criteria--such as exactness, coverage, topicality, and contextual fit--to improve the robustness and interpretability of retrieval evaluations compared to direct grading methods. We validate this approach on three datasets: the TREC Deep Learning tracks from 2019 and 2020, as well as LLMJudge (based on TREC DL 2023). Our results demonstrate that Multi-Criteria judgments enhance the system ranking/leaderboard performance. Moreover, we highlight the strengths and limitations of this approach relative to direct grading approaches, offering insights that can guide the development of future automatic evaluation frameworks in information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09483v1">Does UMBRELA Work on Other LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 9 pages, 2 figures, accepted to SIGIR 2025
    </div>
    <details class="paper-abstract">
      We reproduce the UMBRELA LLM Judge evaluation framework across a range of large language models (LLMs) to assess its generalizability beyond the original study. Our investigation evaluates how LLM choice affects relevance assessment accuracy, focusing on leaderboard rank correlation and per-label agreement metrics. Results demonstrate that UMBRELA with DeepSeek V3 obtains very comparable performance to GPT-4o (used in original work). For LLaMA-3.3-70B we obtain slightly lower performance, which further degrades with smaller LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09481v1">Evaluating LLMs on Sequential API Call Through Automated Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09477v1">Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 submitted to ARR May
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at https://github.com/DavidZWZ/Awesome-RAG-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00648v3">DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepeted bythe 2nd Conference on Language Modeling (COLM 2025). Source code \url{https://github.com/JingyangXiang/DFRot}
    </div>
    <details class="paper-abstract">
      Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomenon remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, Outlier-Free and Massive Activation-Free, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-70B, a model known for its quantization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10620v1">LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepted at SSTD 2025 (Tutorial). arXiv admin note: text overlap with arXiv:2505.02583
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a promising paradigm for time series analytics, leveraging their massive parameters and the shared sequential nature of textual and time series data. However, a cross-modality gap exists between time series and textual data, as LLMs are pre-trained on textual corpora and are not inherently optimized for time series. In this tutorial, we provide an up-to-date overview of LLM-based cross-modal time series analytics. We introduce a taxonomy that classifies existing approaches into three groups based on cross-modal modeling strategies, e.g., conversion, alignment, and fusion, and then discuss their applications across a range of downstream tasks. In addition, we summarize several open challenges. This tutorial aims to expand the practical application of LLMs in solving real-world problems in cross-modal time series analytics while balancing effectiveness and efficiency. Participants will gain a thorough understanding of current advancements, methodologies, and future research directions in cross-modal time series analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10605v1">RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09407v1">LLM-Stackelberg Games: Conjectural Reasoning Equilibria and Their Applications to Spearphishing</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      We introduce the framework of LLM-Stackelberg games, a class of sequential decision-making models that integrate large language models (LLMs) into strategic interactions between a leader and a follower. Departing from classical Stackelberg assumptions of complete information and rational agents, our formulation allows each agent to reason through structured prompts, generate probabilistic behaviors via LLMs, and adapt their strategies through internal cognition and belief updates. We define two equilibrium concepts: reasoning and behavioral equilibrium, which aligns an agent's internal prompt-based reasoning with observable behavior, and conjectural reasoning equilibrium, which accounts for epistemic uncertainty through parameterized models over an opponent's response. These layered constructs capture bounded rationality, asymmetric information, and meta-cognitive adaptation. We illustrate the framework through a spearphishing case study, where a sender and a recipient engage in a deception game using structured reasoning prompts. This example highlights the cognitive richness and adversarial potential of LLM-mediated interactions. Our results show that LLM-Stackelberg games provide a powerful paradigm for modeling decision-making in domains such as cybersecurity, misinformation, and recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22362v2">Supposedly Equivalent Facts That Aren't? Entity Frequency in Pre-training Induces Asymmetry in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Understanding and mitigating hallucinations in Large Language Models (LLMs) is crucial for ensuring reliable content generation. While previous research has primarily focused on "when" LLMs hallucinate, our work explains "why" and directly links model behaviour to the pre-training data that forms their prior knowledge. Specifically, we demonstrate that an asymmetry exists in the recognition of logically equivalent facts, which can be attributed to frequency discrepancies of entities appearing as subjects versus objects. Given that most pre-training datasets are inaccessible, we leverage the fully open-source OLMo series by indexing its Dolma dataset to estimate entity frequencies. Using relational facts (represented as triples) from Wikidata5M, we construct probing datasets to isolate this effect. Our experiments reveal that facts with a high-frequency subject and a low-frequency object are better recognised than their inverse, despite their logical equivalence. The pattern reverses in low-to-high frequency settings, and no statistically significant asymmetry emerges when both entities are high-frequency. These findings highlight the influential role of pre-training data in shaping model predictions and provide insights for inferring the characteristics of pre-training data in closed or partially closed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00555v2">Prune 'n Predict: Optimizing LLM Decision-making with Conformal Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are empowering decision-making in several applications, including tool or API usage and answering multiple-choice questions (MCQs). However, incorrect outputs pose significant risks in high-stakes domains like healthcare and finance. To quantify LLM uncertainty and thereby mitigate these risks, recent works employ conformal prediction (CP), a model- and distribution-agnostic framework that uses LLM outputs to generate a \emph{prediction set} containing the true answer with high probability. Leveraging CP, we propose \emph{conformal revision of questions} (CROQ), which revises the question by narrowing down the available choices to those in the prediction set and asking the LLM the revised question. We expect LLMs to be more accurate on revised questions with fewer choices. Furthermore, we expect CROQ to be effective when the prediction sets from CP are small. Commonly used logit scores often lead to large sets, diminishing CROQ's effectiveness. To overcome this, we propose CP-OPT, an optimization framework to learn scores that minimize set sizes while maintaining coverage. Our extensive experiments on MMLU, ToolAlpaca, and TruthfulQA datasets with multiple LLMs show that CROQ improves accuracy over the standard inference, with more pronounced gains when paired with CP-OPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23377v2">Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 7 pages, 5 main pages of text, 5 figures, 2 tables. Research work performed at CACI INTL INC
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09329v1">When Developer Aid Becomes Security Debt: A Systematic Analysis of Insecure Behaviors in LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      LLM-based coding agents are rapidly being deployed in software development, yet their security implications remain poorly understood. These agents, while capable of accelerating software development, may inadvertently introduce insecure practices. We conducted the first systematic security evaluation of autonomous coding agents, analyzing over 12,000 actions across five state-of-the-art models (GPT-4o, GPT-4.1, Claude variants) on 93 real-world software setup tasks. Our findings reveal significant security concerns: 21% of agent trajectories contained insecure actions, with models showing substantial variation in security behavior. We developed a high-precision detection system that identified four major vulnerability categories, with information exposure (CWE-200) being the most prevalent one. We also evaluated mitigation strategies including feedback mechanisms and security reminders with various effectiveness between models. GPT-4.1 demonstrated exceptional security awareness with 96.8% mitigation success. Our work provides the first comprehensive framework for evaluating coding agent security and highlights the need for security-aware design of next generation LLM-based coding agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23978v2">LLM Agents Are the Antidote to Walled Gardens</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      While the Internet's core infrastructure was designed to be open and universal, today's application layer is dominated by closed, proprietary platforms. Open and interoperable APIs require significant investment, and market leaders have little incentive to enable data exchange that could erode their user lock-in. We argue that LLM-based agents fundamentally disrupt this status quo. Agents can automatically translate between data formats and interact with interfaces designed for humans: this makes interoperability dramatically cheaper and effectively unavoidable. We name this shift universal interoperability: the ability for any two digital services to exchange data seamlessly using AI-mediated adapters. Universal interoperability undermines monopolistic behaviours and promotes data portability. However, it can also lead to new security risks and technical debt. Our position is that the ML community should embrace this development while building the appropriate frameworks to mitigate the downsides. By acting now, we can harness AI to restore user freedom and competitive markets without sacrificing security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13554v2">LLMs' Leaning in European Elections</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Many studies suggest that LLMs have left wing leans. The article extends previous analysis of US presidential elections considering several virtual elections in multiple European countries. The analysis considers multiple LLMs and the results confirm the extent of the leaning. Furthermore, the results show that the leaning is not uniform between countries. Sometimes, models refuse to take a position in the virtual elections, but the refusal rate itself is not uniform between countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09255v1">StockSim: A Dual-Mode Order-Level Simulator for Evaluating Multi-Agent LLMs in Financial Markets</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      We present StockSim, an open-source simulation platform for systematic evaluation of large language models (LLMs) in realistic financial decision-making scenarios. Unlike previous toolkits that offer limited scope, StockSim delivers a comprehensive system that fully models market dynamics and supports diverse simulation modes of varying granularity. It incorporates critical real-world factors, such as latency, slippage, and order-book microstructure, that were previously neglected, enabling more faithful and insightful assessment of LLM-based trading agents. An extensible, role-based agent framework supports heterogeneous trading strategies and multi-agent coordination, making StockSim a uniquely capable testbed for NLP research on reasoning under uncertainty and sequential decision-making. We open-source all our code at https: //github.com/harrypapa2002/StockSim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07186v2">Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 CoLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit cognitive biases -- systematic tendencies of irrational decision-making, similar to those seen in humans. Prior work has found that these biases vary across models and can be amplified by instruction tuning. However, it remains unclear if these differences in biases stem from pretraining, finetuning, or even random noise due to training stochasticity. We propose a two-step causal experimental approach to disentangle these factors. First, we finetune models multiple times using different random seeds to study how training randomness affects over $30$ cognitive biases. Second, we introduce \emph{cross-tuning} -- swapping instruction datasets between models to isolate bias sources. This swap uses datasets that led to different bias patterns, directly testing whether biases are dataset-dependent. Our findings reveal that while training randomness introduces some variability, biases are mainly shaped by pretraining: models with the same pretrained backbone exhibit more similar bias patterns than those sharing only finetuning data. These insights suggest that understanding biases in finetuned models requires considering their pretraining origins beyond finetuning effects. This perspective can guide future efforts to develop principled strategies for evaluating and mitigating bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23502v2">LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Driven by large-scale contrastive vision-language pre-trained models such as CLIP, recent advancements in the image-text matching task have achieved remarkable success in representation learning. Due to image-level visual-language alignment, CLIP falls short in understanding fine-grained details such as object attributes and spatial relationships between objects. Recent efforts have attempted to compel CLIP to acquire structured visual representations by introducing prompt learning to achieve object-level alignment. While achieving promising results, they still lack the capability to perceive actions, which are crucial for describing the states or relationships between objects. Therefore, we propose to endow CLIP with fine-grained action-level understanding by introducing an LLM-enhanced action-aware multi-modal prompt-tuning method, incorporating the action-related external knowledge generated by large language models (LLMs). Specifically, we design an action triplet prompt and an action state prompt to exploit compositional semantic knowledge and state-related causal knowledge implicitly stored in LLMs. Subsequently, we propose an adaptive interaction module to aggregate attentive visual features conditioned on action-aware prompted knowledge for establishing discriminative and action-aware visual representations, which further improves the performance. Comprehensive experimental results on two benchmark datasets demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09135v1">Position Paper: Programming Language Techniques for Bridging LLM Code Generation Semantic Gaps</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated remarkable capabilities in automated code generation, yet their statistical nature and black-box characteristics create significant semantic gaps manifested through syntax errors, semantic hallucinations, and reliability concerns. This position paper argues that principled integration of Programming Language (PL) techniques is essential for bridging these gaps. Through structured program representations, formal correctness guarantees, and robust verification mechanisms, PL techniques can elevate LLM-generated code from statistical pattern matching to truly reliable and trustworthy levels. This integration is crucial for developing systems that generate code that is not only functionally correct but also interpretable, verifiable, and ultimately trustworthy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v1">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12318v2">UTF:Undertrained Tokens as Fingerprints A Novel Approach to LLM Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Fingerprinting large language models (LLMs) is essential for verifying model ownership, ensuring authenticity, and preventing misuse. Traditional fingerprinting methods often require significant computational overhead or white-box verification access. In this paper, we introduce UTF, a novel and efficient approach to fingerprinting LLMs by leveraging under-trained tokens. Under-trained tokens are tokens that the model has not fully learned during its training phase. By utilizing these tokens, we perform supervised fine-tuning to embed specific input-output pairs into the model. This process allows the LLM to produce predetermined outputs when presented with certain inputs, effectively embedding a unique fingerprint. Our method has minimal overhead and impact on model's performance, and does not require white-box access to target model's ownership identification. Compared to existing fingerprinting methods, UTF is also more effective and robust to fine-tuning and random guess.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09750v2">Pinning "Reflection" on the Agenda: Investigating Reflection in Human-LLM Co-Creation for Creative Coding</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 6 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into creative coding, yet how users reflect, and how different co-creation conditions influence reflective behavior, remains underexplored. This study investigates situated, moment-to-moment reflection in creative coding under two prompting strategies: the entire task invocation (T1) and decomposed subtask invocation (T2), to examine their effects on reflective behavior. Our mixed-method results reveal three distinct reflection types and show that T2 encourages more frequent, strategic, and generative reflection, fostering diagnostic reasoning and goal redefinition. These findings offer insights into how LLM-based tools foster deeper creative engagement through structured, behaviorally grounded reflection support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10596v1">PLEX: Perturbation-free Local Explanations for LLM-Based Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in text classification, but their complexity hinders interpretability, making it difficult to understand the reasoning behind their predictions. Explainable AI (XAI) methods like LIME and SHAP offer local explanations by identifying influential words, but they rely on computationally expensive perturbations. These methods typically generate thousands of perturbed sentences and perform inferences on each, incurring a substantial computational burden, especially with LLMs. To address this, we propose \underline{P}erturbation-free \underline{L}ocal \underline{Ex}planation (PLEX), a novel method that leverages the contextual embeddings extracted from the LLM and a ``Siamese network" style neural network trained to align with feature importance scores. This one-off training eliminates the need for subsequent perturbations, enabling efficient explanations for any new sentence. We demonstrate PLEX's effectiveness on four different classification tasks (sentiment, fake news, fake COVID-19 news and depression), showing more than 92\% agreement with LIME and SHAP. Our evaluation using a ``stress test" reveals that PLEX accurately identifies influential words, leading to a similar decline in classification accuracy as observed with LIME and SHAP when these words are removed. Notably, in some cases, PLEX demonstrates superior performance in capturing the impact of key features. PLEX dramatically accelerates explanation, reducing time and computational overhead by two and four orders of magnitude, respectively. This work offers a promising solution for explainable LLM-based text classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08794v1">One Token to Fool LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Generative reward models (also known as LLMs-as-judges), which use large language models (LLMs) to evaluate answer quality, are increasingly adopted in reinforcement learning with verifiable rewards (RLVR). They are often preferred over rigid rule-based metrics, especially for complex reasoning tasks involving free-form outputs. In this paradigm, an LLM is typically prompted to compare a candidate answer against a ground-truth reference and assign a binary reward indicating correctness. Despite the seeming simplicity of this comparison task, we find that generative reward models exhibit surprising vulnerabilities to superficial manipulations: non-word symbols (e.g., ":" or ".") or reasoning openers like "Thought process:" and "Let's solve this problem step by step." can often lead to false positive rewards. We demonstrate that this weakness is widespread across LLMs, datasets, and prompt formats, posing a serious threat for core algorithmic paradigms that rely on generative reward models, such as rejection sampling, preference optimization, and RLVR. To mitigate this issue, we introduce a simple yet effective data augmentation strategy and train a new generative reward model with substantially improved robustness. Our findings highlight the urgent need for more reliable LLM-based evaluation methods. We release our robust, general-domain reward model and its synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08671v1">LLMCup: Ranking-Enhanced Comment Updating with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While comments are essential for enhancing code readability and maintainability in modern software projects, developers are often motivated to update code but not comments, leading to outdated or inconsistent documentation that hinders future understanding and maintenance. Recent approaches such as CUP and HebCup have attempted automatic comment updating using neural sequence-to-sequence models and heuristic rules, respectively. However, these methods can miss or misinterpret crucial information during comment updating, resulting in inaccurate comments, and they often struggle with complex update scenarios. Given these challenges, a promising direction lies in leveraging large language models (LLMs), which have shown impressive performance in software engineering tasks such as comment generation, code synthesis, and program repair. This suggests their strong potential to capture the logic behind code modifications - an ability that is crucial for the task of comment updating. Nevertheless, selecting an appropriate prompt strategy for an LLM on each update case remains challenging. To address this, we propose a novel comment updating framework, LLMCup, which first uses multiple prompt strategies to provide diverse candidate updated comments via an LLM, and then employs a ranking model, CupRank, to select the best candidate as final updated comment. Experimental results demonstrate the effectiveness of LLMCup, with improvements over state-of-the-art baselines (CUP and HebCup) by 49.0%-116.9% in Accuracy, 10.8%-20% in BLEU-4, 4.6% in METEOR, 0.9%-1.9% in F1, and 2.1%-3.4% in SentenceBert similarity. Furthermore, a user study shows that comments updated by LLMCup sometimes surpass human-written updates, highlighting the importance of incorporating human evaluation in comment quality assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08627v1">NL in the Middle: Code Translation with LLMs and Intermediate Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Studies show that large language models (LLMs) produce buggy code translations. One avenue to improve translation accuracy is through intermediate representations, which could provide structured insights to guide the model's understanding. We explore whether code translation using LLMs can benefit from intermediate representations via natural language (NL) and abstract syntax trees (ASTs). Since prompt engineering greatly affects LLM performance, we consider several ways to integrate these representations, from one-shot to chain-of-thought (CoT) prompting. Using Open Gpt4 8X7B and specialized StarCoder and CodeGen models on popular code translation benchmarks (CodeNet and AVATAR), we find that CoT with an intermediate NL summary performs best, with an increase of 13.8% and 6.7%, respectively, in successful translations for the best-performing model (Open Gpt4 8X7B) compared to the zero-shot prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08621v1">A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08311v2">Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Pol G. Recasens, Ferran Agullo: equal contribution. Paper accepted at IEEE CLOUD 2025
    </div>
    <details class="paper-abstract">
      Large language models have been widely adopted across different tasks, but their auto-regressive generation nature often leads to inefficient resource utilization during inference. While batching is commonly used to increase throughput, performance gains plateau beyond a certain batch size, especially with smaller models, a phenomenon that existing literature typically explains as a shift to the compute-bound regime. In this paper, through an in-depth GPU-level analysis, we reveal that large-batch inference remains memory-bound, with most GPU compute capabilities underutilized due to DRAM bandwidth saturation as the primary bottleneck. To address this, we propose a Batching Configuration Advisor (BCA) that optimizes memory allocation, reducing GPU memory requirements with minimal impact on throughput. The freed memory and underutilized GPU compute capabilities can then be leveraged by concurrent workloads. Specifically, we use model replication to improve serving throughput and GPU utilization. Our findings challenge conventional assumptions about LLM inference, offering new insights and practical strategies for improving resource utilization, particularly for smaller language models. The code is publicly available at https://github.com/FerranAgulloLopez/vLLMBatchingMemoryGap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08616v1">AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large-language models (LLMs) have demonstrated powerful problem-solving capabilities, in particular when organized in multi-agent systems. However, the advent of such systems also raises several questions on the ability of a complex network of agents to effectively self-organize and collaborate. While measuring performance on standard reasoning benchmarks indicates how well multi-agent systems can solve reasoning tasks, it is unclear whether these systems are able to leverage their topology effectively. Here, we propose AgentsNet, a new benchmark for multi-agent reasoning. By drawing inspiration from classical problems in distributed systems and graph theory, AgentsNet measures the ability of multi-agent systems to collaboratively form strategies for problem-solving, self-organization, and effective communication given a network topology. We evaluate a variety of baseline methods on AgentsNet including homogeneous networks of agents which first have to agree on basic protocols for organization and communication. We find that some frontier LLMs are already demonstrating strong performance for small networks but begin to fall off once the size of the network scales. While existing multi-agent benchmarks cover at most 2-5 agents, AgentsNet is practically unlimited in size and can scale with new generations of LLMs. As such, we also probe frontier models in a setup with up to 100 agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08538v1">The AI Language Proficiency Monitor -- Tracking the Progress of LLMs on Multilingual Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      To ensure equitable access to the benefits of large language models (LLMs), it is essential to evaluate their capabilities across the world's languages. We introduce the AI Language Proficiency Monitor, a comprehensive multilingual benchmark that systematically assesses LLM performance across up to 200 languages, with a particular focus on low-resource languages. Our benchmark aggregates diverse tasks including translation, question answering, math, and reasoning, using datasets such as FLORES+, MMLU, GSM8K, TruthfulQA, and ARC. We provide an open-source, auto-updating leaderboard and dashboard that supports researchers, developers, and policymakers in identifying strengths and gaps in model performance. In addition to ranking models, the platform offers descriptive insights such as a global proficiency map and trends over time. By complementing and extending prior multilingual benchmarks, our work aims to foster transparency, inclusivity, and progress in multilingual AI. The system is available at https://huggingface.co/spaces/fair-forward/evals-for-every-language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08523v1">InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: (1) the privacy risks associated with commercial LLMs, driving the adoption of local deployment, and (2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system. In this study, we present InferLog, the first LLM inference optimization method for online log parsing. Our key insight is that the inference efficiency emerges as the vital bottleneck in LLM-based online log parsing, rather than parsing accuracy. InferLog accelerates inference by designing (1) A Prefix-aware ICL Refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. (2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-related configuration for dynamic log parsing workloads. The experimental results based on Loghub dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08513v1">Advancing Multimodal LLMs by Large-Scale 3D Visual Instruction Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) struggle with accurately capturing camera-object relations, especially for object orientation, camera viewpoint, and camera shots. This stems from the fact that existing MLLMs are trained on images with limited diverse camera-object relations and corresponding textual descriptions. To address this, we propose a synthetic generation pipeline to create large-scale 3D visual instruction datasets. Our framework takes 3D assets as input and uses rendering and diffusion-based image generation models to create photorealistic images preserving precise camera-object relations. Additionally, large language models (LLMs) are used to generate text prompts for guiding visual instruction tuning and controlling image generation. We create Ultimate3D, a dataset of 240K VQAs with precise camera-object annotations, and corresponding benchmark. MLLMs fine-tuned on our proposed dataset outperform commercial models by a large margin, achieving an average accuracy improvement of 33.4% on camera-object relation recognition tasks. Our code, dataset, and benchmark will contribute to broad MLLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08498v1">Semantic-Augmented Latent Topic Modeling with LLM-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Latent Dirichlet Allocation (LDA) is a prominent generative probabilistic model used for uncovering abstract topics within document collections. In this paper, we explore the effectiveness of augmenting topic models with Large Language Models (LLMs) through integration into two key phases: Initialization and Post-Correction. Since the LDA is highly dependent on the quality of its initialization, we conduct extensive experiments on the LLM-guided topic clustering for initializing the Gibbs sampling algorithm. Interestingly, the experimental results reveal that while the proposed initialization strategy improves the early iterations of LDA, it has no effect on the convergence and yields the worst performance compared to the baselines. The LLM-enabled post-correction, on the other hand, achieved a promising improvement of 5.86% in the coherence evaluation. These results highlight the practical benefits of the LLM-in-the-loop approach and challenge the belief that LLMs are always the superior text mining alternative.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08491v1">A Third Paradigm for LLM Evaluation: Dialogue Game-Based Evaluation using clembench</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 All code required to run the benchmark, as well as extensive documentation, is available at https://github.com/clembench/clembench
    </div>
    <details class="paper-abstract">
      There are currently two main paradigms for evaluating large language models (LLMs), reference-based evaluation and preference-based evaluation. The first, carried over from the evaluation of machine learning models in general, relies on pre-defined task instances, for which reference task executions are available. The second, best exemplified by the LM-arena, relies on (often self-selected) users bringing their own intents to a site that routes these to several models in parallel, among whose responses the user then selects their most preferred one. The former paradigm hence excels at control over what is tested, while the latter comes with higher ecological validity, testing actual use cases interactively. Recently, a third complementary paradigm has emerged that combines some of the strengths of these approaches, offering control over multi-turn, reference-free, repeatable interactions, while stressing goal-directedness: dialogue game based evaluation. While the utility of this approach has been shown by several projects, its adoption has been held back by the lack of a mature, easily re-usable implementation. In this paper, we present clembench, which has been in continuous development since 2023 and has in its latest release been optimized for ease of general use. We describe how it can be used to benchmark one's own models (using a provided set of benchmark game instances in English), as well as how easily the benchmark itself can be extended with new, tailor-made targeted tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08472v1">Pre-Training LLMs on a budget: A comparison of three optimizers</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Optimizers play a decisive role in reducing pre-training times for LLMs and achieving better-performing models. In this study, we compare three major variants: the de-facto standard AdamW, the simpler Lion, developed through an evolutionary search, and the second-order optimizer Sophia. For better generalization, we train with two different base architectures and use a single- and a multiple-epoch approach while keeping the number of tokens constant. Using the Maximal Update Parametrization and smaller proxy models, we tune relevant hyperparameters separately for each combination of base architecture and optimizer. We found that while the results from all three optimizers were in approximately the same range, Sophia exhibited the lowest training and validation loss, Lion was fastest in terms of training GPU hours but AdamW led to the best downstream evaluation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v3">The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08427v1">ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      Current knowledge editing methods for large language models (LLMs) struggle to maintain logical consistency when propagating ripple effects to associated facts. We propose ChainEdit, a framework that synergizes knowledge graph-derived logical rules with LLM logical reasoning capabilities to enable systematic chain updates. By automatically extracting logical patterns from structured knowledge bases and aligning them with LLMs' internal logics, ChainEdit dynamically generates and edits logically connected knowledge clusters. Experiments demonstrate an improvement of more than 30% in logical generalization over baselines while preserving editing reliability and specificity. We further address evaluation biases in existing benchmarks through knowledge-aware protocols that disentangle external dependencies. This work establishes new state-of-the-art performance on ripple effect while ensuring internal logical consistency after knowledge editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08392v1">Multi-Agent LLMs as Ethics Advocates in AI-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08350v1">Exploring Design of Multi-Agent LLM Dialogues for Research Ideation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 16 pages, 1 figure, appendix. Accepted to SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support creative tasks such as research idea generation. While recent work has shown that structured dialogues between LLMs can improve the novelty and feasibility of generated ideas, the optimal design of such interactions remains unclear. In this study, we conduct a comprehensive analysis of multi-agent LLM dialogues for scientific ideation. We compare different configurations of agent roles, number of agents, and dialogue depth to understand how these factors influence the novelty and feasibility of generated ideas. Our experimental setup includes settings where one agent generates ideas and another critiques them, enabling iterative improvement. Our results show that enlarging the agent cohort, deepening the interaction depth, and broadening agent persona heterogeneity each enrich the diversity of generated ideas. Moreover, specifically increasing critic-side diversity within the ideation-critique-revision loop further boosts the feasibility of the final proposals. Our findings offer practical guidelines for building effective multi-agent LLM systems for scientific ideation. Our code is available at https://github.com/g6000/MultiAgent-Research-Ideator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08339v1">What Factors Affect LLMs and RLLMs in Financial Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v4">Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08325v1">CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08267v1">A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Presented at ICML 2025 Workshop on The second AI for MATH
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning of Large Language Models (LLMs) is a pivotal challenge in advancing AI capabilities. While Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are the dominant training paradigms, a systematic methodology for combining them to maximize both accuracy and efficiency remains largely unexplored. This paper introduces a practical and effective training recipe that strategically integrates extended SFT with RL from online inference (GRPO). We posit that these methods play complementary, not competing, roles: a prolonged SFT phase first pushes the model's accuracy to its limits, after which a GRPO phase dramatically improves token efficiency while preserving this peak performance. Our experiments reveal that extending SFT for as many as 10 epochs is crucial for performance breakthroughs, and that the primary role of GRPO in this framework is to optimize solution length. The efficacy of our recipe is rigorously validated through top-tier performance on challenging benchmarks, including a high rank among over 2,200 teams in the strictly leak-free AI Mathematical Olympiad (AIMO). This work provides the community with a battle-tested blueprint for developing state-of-the-art mathematical reasoners that are both exceptionally accurate and practically efficient. To ensure full reproducibility and empower future research, we will open-source our entire framework, including all code, model checkpoints, and training configurations at https://github.com/analokmaus/kaggle-aimo2-fast-math-r1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17621v3">Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a pivotal method for improving the reasoning capabilities of Large Language Models (LLMs). However, prevalent RL approaches such as Proximal Policy Optimization (PPO) and Group-Regularized Policy Optimization (GRPO) face critical limitations due to their reliance on sparse outcome-based rewards and inadequate mechanisms for incentivizing exploration. These limitations result in inefficient guidance for multi-step reasoning processes. Specifically, sparse reward signals fail to deliver effective or sufficient feedback, particularly for challenging problems. Furthermore, such reward structures induce systematic biases that prioritize exploitation of familiar trajectories over novel solution discovery. These shortcomings critically hinder performance in complex reasoning tasks, which inherently demand iterative refinement across ipntermediate steps. To address these challenges, we propose an Intrinsic Motivation guidEd exploratioN meThOd foR LLM Reasoning (i-MENTOR), a novel method designed to both deliver dense rewards and amplify explorations in the RL-based training paradigm. i-MENTOR introduces three key innovations: trajectory-aware exploration rewards that mitigate bias in token-level strategies while maintaining computational efficiency; dynamic reward scaling to stabilize exploration and exploitation in large action spaces; and advantage-preserving reward implementation that maintains advantage distribution integrity while incorporating exploratory guidance. Experiments across three public datasets demonstrate i-MENTOR's effectiveness with a 22.39% improvement on the difficult dataset Countdown-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08255v1">Quantum-Accelerated Neural Imputation with Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Missing data presents a critical challenge in real-world datasets, significantly degrading the performance of machine learning models. While Large Language Models (LLMs) have recently demonstrated remarkable capabilities in tabular data imputation, exemplified by frameworks like UnIMP, their reliance on classical embedding methods often limits their ability to capture complex, non-linear correlations, particularly in mixed-type data scenarios encompassing numerical, categorical, and textual features. This paper introduces Quantum-UnIMP, a novel framework that integrates shallow quantum circuits into an LLM-based imputation architecture. Our core innovation lies in replacing conventional classical input embeddings with quantum feature maps generated by an Instantaneous Quantum Polynomial (IQP) circuit. This approach enables the model to leverage quantum phenomena such as superposition and entanglement, thereby learning richer, more expressive representations of data and enhancing the recovery of intricate missingness patterns. Our experiments on benchmark mixed-type datasets demonstrate that Quantum-UnIMP reduces imputation error by up to 15.2% for numerical features (RMSE) and improves classification accuracy by 8.7% for categorical features (F1-Score) compared to state-of-the-art classical and LLM-based methods. These compelling results underscore the profound potential of quantum-enhanced representations for complex data imputation tasks, even with near-term quantum hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08235v1">InsightBuild: LLM-Powered Causal Reasoning in Smart Building Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Smart buildings generate vast streams of sensor and control data, but facility managers often lack clear explanations for anomalous energy usage. We propose InsightBuild, a two-stage framework that integrates causality analysis with a fine-tuned large language model (LLM) to provide human-readable, causal explanations of energy consumption patterns. First, a lightweight causal inference module applies Granger causality tests and structural causal discovery on building telemetry (e.g., temperature, HVAC settings, occupancy) drawn from Google Smart Buildings and Berkeley Office datasets. Next, an LLM, fine-tuned on aligned pairs of sensor-level causes and textual explanations, receives as input the detected causal relations and generates concise, actionable explanations. We evaluate InsightBuild on two real-world datasets (Google: 2017-2022; Berkeley: 2018-2020), using expert-annotated ground-truth causes for a held-out set of anomalies. Our results demonstrate that combining explicit causal discovery with LLM-based natural language generation yields clear, precise explanations that assist facility managers in diagnosing and mitigating energy inefficiencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08232v1">Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Accepted to the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), co-located with ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxy students in the development of Intelligent Tutoring Systems (ITSs) and in piloting test questions. However, to what extent these proxy students accurately emulate the behavior and characteristics of real students remains an open question. To investigate this, we collected a dataset of 489 items from the National Assessment of Educational Progress (NAEP), covering mathematics and reading comprehension in grades 4, 8, and 12. We then apply an Item Response Theory (IRT) model to position 11 diverse and state-of-the-art LLMs on the same ability scale as real student populations. Our findings reveal that, without guidance, strong general-purpose models consistently outperform the average student at every grade, while weaker or domain-mismatched models may align incidentally. Using grade-enforcement prompts changes models' performance, but whether they align with the average grade-level student remains highly model- and prompt-specific: no evaluated model-prompt pair fits the bill across subjects and grades, underscoring the need for new training and evaluation strategies. We conclude by providing guidelines for the selection of viable proxies based on our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09037v1">ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 10 pages total (including appendix), ICML 2025 Workshop on Reliable and Responsible Foundation Models
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used as decision aids. However, users have diverse values and preferences that can affect their decision-making, which requires novel methods for LLM alignment and personalization. Existing LLM comparison tools largely focus on benchmarking tasks, such as knowledge-based question answering. In contrast, our proposed ALIGN system focuses on dynamic personalization of LLM-based decision-makers through prompt-based alignment to a set of fine-grained attributes. Key features of our system include robust configuration management, structured output generation with reasoning, and several algorithm implementations with swappable LLM backbones, enabling different types of analyses. Our user interface enables a qualitative, side-by-side comparison of LLMs and their alignment to various attributes, with a modular backend for easy algorithm integration. Additionally, we perform a quantitative analysis comparing alignment approaches in two different domains: demographic alignment for public opinion surveys and value alignment for medical triage decision-making. The entire ALIGN framework is open source and will enable new research on reliable, responsible, and personalized LLM-based decision-makers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09019v1">On Evaluating Performance of LLM Inference Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Model (LLM) inference systems has yielded significant efficiency improvements. However, our systematic analysis reveals that current evaluation methodologies frequently exhibit fundamental flaws, often manifesting as common evaluation anti-patterns that obscure true performance characteristics and impede scientific progress. Through a comprehensive examination of recent systems, we identify recurring anti-patterns across three key dimensions: Baseline Fairness, Evaluation Setup, and Metric Design. These anti-patterns are uniquely problematic for LLM inference due to its dual-phase nature combining distinct prefill and decode operations, its handling of highly heterogeneous workloads, and its strict temporal requirements for interactive use. We demonstrate how common anti-patterns -- such as inadequate baseline comparisons that conflate engineering effort with algorithmic novelty, workload selections that fail to represent production scenarios, and metric normalizations that hide substantial performance variability like generation stalls-lead to misleading conclusions. To address these challenges, we provide a comprehensive checklist derived from our analysis, establishing a framework for recognizing and avoiding these anti-patterns in favor of robust LLM inference evaluation. To demonstrate the practical application of our framework, we present a case study analyzing speculative decoding, a technique whose bursty, non-uniform token generation is easily misinterpreted when evaluated using approaches characteristic of these anti-patterns. Our work establishes a rigorous foundation for evaluation methodology, enabling meaningful comparisons, ensuring reproducible results, and ultimately accelerating genuine progress in LLM inference systems by moving beyond common anti-patterns to align evaluation with real-world requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08979v1">PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      We introduce Projection-based Reduction of Implicit Spurious bias in vision-language Models (PRISM), a new data-free and task-agnostic solution for bias mitigation in VLMs like CLIP. VLMs often inherit and amplify biases in their training data, leading to skewed predictions. PRISM is designed to debias VLMs without relying on predefined bias categories or additional external data. It operates in two stages: first, an LLM is prompted with simple class prompts to generate scene descriptions that contain spurious correlations. Next, PRISM uses our novel contrastive-style debiasing loss to learn a projection that maps the embeddings onto a latent space that minimizes spurious correlations while preserving the alignment between image and text embeddings.Extensive experiments demonstrate that PRISM outperforms current debiasing methods on the commonly used Waterbirds and CelebA datasets We make our code public at: https://github.com/MahdiyarMM/PRISM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08960v1">How to Train a Leader: Hierarchical Reasoning in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved strong performance on a wide range of complex reasoning tasks, yet further gains are often possible by leveraging the complementary strengths of multiple models. While multi-agent frameworks can improve solution quality by leveraging multiple LLMs, existing methods are often computationally expensive, both at training and inference time. In this work, we introduce a hierarchical multi-agent framework that addresses these challenges by training only a single leader LLM to coordinate a team of untrained peer agents. To this end, we propose Multi-agent guided Leader Policy \textbf{O}ptimization (MLPO), a novel approach which trains the leader to evaluate and synthesize agent responses without auxiliary value networks or explicit agent feedback. Leaders trained with MLPO exhibit improved performance not only when interacting with the agent team at inference time, but also enjoy improved performance when deployed in single-agent settings without the team. Empirical results on Big-Bench Hard (BBH), MATH, and MMLU demonstrate that our framework achieves substantial performance improvements over both single-agent and multi-agent baselines. Our results highlight the effectiveness and efficiency of training a single, flexible leader for collaborative reasoning in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v2">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08944v1">Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 ICML 2025 Workshop on MAS
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems have demonstrated remarkable promise for tackling complex tasks by breaking them down into subtasks that are iteratively planned, executed, observed, and refined. Despite their effectiveness, these systems often incur high latency because real-world problems frequently demand multiple iterative cycles of reasoning steps. To address this challenge, we propose M1-Parallel, a framework that concurrently runs multiple multi-agent teams in parallel to uncover distinct solution paths. By leveraging an event-driven communication model with asynchronous messaging, M1-Parallel efficiently capitalizes on the inherent diversity of valid plans to either reduce end-to-end latency or boost task completion rates. Our experiments on complex tasks show that M1-Parallel with early termination achieves up to $2.2\times$ speedup while preserving accuracy, and that M1-Parallel with aggregation yields higher task completion rates. We further investigate strategies aimed at encouraging diverse execution plans but observe no additional performance gains over repeated sampling. Overall, these findings underscore the potential of parallel plan execution for optimizing multi-agent systems for real-world, high-complexity reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08924v1">From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) requires robust benchmarks that encompass not only academic domains but also industrial fields to effectively evaluate their applicability in real-world scenarios. In this paper, we introduce two Korean expert-level benchmarks. KMMLU-Redux, reconstructed from the existing KMMLU, consists of questions from the Korean National Technical Qualification exams, with critical errors removed to enhance reliability. KMMLU-Pro is based on Korean National Professional Licensure exams to reflect professional knowledge in Korea. Our experiments demonstrate that these benchmarks comprehensively represent industrial knowledge in Korea. We release our dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08916v1">Evaluating LLMs in Medicine: A Call for Rigor, Transparency</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Objectives: To evaluate the current limitations of large language models (LLMs) in medical question answering, focusing on the quality of datasets used for their evaluation. Materials and Methods: Widely-used benchmark datasets, including MedQA, MedMCQA, PubMedQA, and MMLU, were reviewed for their rigor, transparency, and relevance to clinical scenarios. Alternatives, such as challenge questions in medical journals, were also analyzed to identify their potential as unbiased evaluation tools. Results: Most existing datasets lack clinical realism, transparency, and robust validation processes. Publicly available challenge questions offer some benefits but are limited by their small size, narrow scope, and exposure to LLM training. These gaps highlight the need for secure, comprehensive, and representative datasets. Conclusion: A standardized framework is critical for evaluating LLMs in medicine. Collaborative efforts among institutions and policymakers are needed to ensure datasets and methodologies are rigorous, unbiased, and reflective of clinical complexities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04219v2">Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their training objectives. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method - Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from the model. Our core idea is to leverage this collapse for unlearning by triggering collapse partially on the sensitive data. We theoretically analyze that our approach converges to the desired outcome, i.e. the LLM unlearns the information in the forget set. We empirically demonstrate that PMC overcomes two key limitations of existing unlearning approaches that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at https://www.cs.cit.tum.de/daml/partial-model-collapse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v1">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Under Review at Information and Software Technology
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10593v1">ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) applications are increasingly relying on external tools to extend their capabilities beyond text generation. However, current tool integration approaches suffer from fragmentation, protocol limitations, and implementation complexity, leading to substantial development overhead. This paper presents Toolregistry, a protocol-agnostic tool management library that simplifies tool registration, representation, execution, and lifecycle management via a unified interface. Our evaluation demonstrates that \toolregistry achieves 60-80% reduction in tool integration code, up to 3.1x performance improvements through concurrent execution, and 100% compatibility with OpenAI function calling standards. Real-world case studies show significant improvements in development efficiency and code maintainability across diverse integration scenarios. \toolregistry is open-source and available at https://github.com/Oaklight/ToolRegistry, with comprehensive documentation at https://toolregistry.readthedocs.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07996v1">Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Can a pretrained neural network adapt its architecture to different inputs without any finetuning? Do we need all layers for simple tasks, and are they adequate for challenging tasks? We found that the layers of a pretrained large language model (LLM) can be manipulated as separate modules to build a better and even shallower model customized for each test sample. In particular, each layer from the pretrained model can be skipped/pruned or repeated multiple times as recurrent neural networks (RNN), and stacked with others in arbitrary orders, yielding a chain-of-layers (CoLa) per sample. This compositional space greatly expands the scope of existing works on looped/recurrent pretrained modules, layer pruning, or early-exit networks. We develop a Monte Carlo Tree Search (MCTS) protocol to explore and identify the optimal CoLa for each sample from math and commonsense reasoning benchmarks. Compared to a static model of a fixed depth, CoLa allows shortcut paths (fast thinking), recurrence of the same layer(s) (slow thinking), and combining both, offering more flexible, dynamic architectures for different inputs. We conduct an extensive analysis of the MCTS-optimized CoLa, which leads to two key findings: (1) For >75% of samples with correct predictions by the original LLM, we can find shorter CoLa, suggesting a large space for improving inference efficiency; (2) For >60% of samples with originally incorrect predictions, we can identify CoLa achieving correct predictions, suggesting a large space of performance enhancement. Our results highlight the shortcomings of using a fixed architecture of pre-trained LLMs for inference on different samples and pave the way to unlock the generalization power of test-time depth adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07990v1">Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Accepted at ICCV2025; Project page: https://www.jshyun.me/projects/sttm
    </div>
    <details class="paper-abstract">
      Video large language models (LLMs) achieve strong video understanding by leveraging a large number of spatio-temporal tokens, but suffer from quadratic computational scaling with token count. To address this, we propose a training-free spatio-temporal token merging method, named STTM. Our key insight is to exploit local spatial and temporal redundancy in video data which has been overlooked in prior work. STTM first transforms each frame into multi-granular spatial tokens using a coarse-to-fine search over a quadtree structure, then performs directed pairwise merging across the temporal dimension. This decomposed merging approach outperforms existing token reduction methods across six video QA benchmarks. Notably, STTM achieves a 2$\times$ speed-up with only a 0.5% accuracy drop under a 50% token budget, and a 3$\times$ speed-up with just a 2% drop under a 30% budget. Moreover, STTM is query-agnostic, allowing KV cache reuse across different questions for the same video. The project page is available at https://www.jshyun.me/projects/sttm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14937v2">Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Transactions of Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07957v1">MIRIX: Multi-Agent Memory System for LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Although memory capabilities of AI agents are gaining increasing attention, existing solutions remain fundamentally limited. Most rely on flat, narrowly scoped memory components, constraining their ability to personalize, abstract, and reliably recall user-specific information over time. To this end, we introduce MIRIX, a modular, multi-agent memory system that redefines the future of AI memory by solving the field's most critical challenge: enabling language models to truly remember. Unlike prior approaches, MIRIX transcends text to embrace rich visual and multimodal experiences, making memory genuinely useful in real-world scenarios. MIRIX consists of six distinct, carefully structured memory types: Core, Episodic, Semantic, Procedural, Resource Memory, and Knowledge Vault, coupled with a multi-agent framework that dynamically controls and coordinates updates and retrieval. This design enables agents to persist, reason over, and accurately retrieve diverse, long-term user data at scale. We validate MIRIX in two demanding settings. First, on ScreenshotVQA, a challenging multimodal benchmark comprising nearly 20,000 high-resolution computer screenshots per sequence, requiring deep contextual understanding and where no existing memory systems can be applied, MIRIX achieves 35% higher accuracy than the RAG baseline while reducing storage requirements by 99.9%. Second, on LOCOMO, a long-form conversation benchmark with single-modal textual input, MIRIX attains state-of-the-art performance of 85.4%, far surpassing existing baselines. These results show that MIRIX sets a new performance standard for memory-augmented LLM agents. To allow users to experience our memory system, we provide a packaged application powered by MIRIX. It monitors the screen in real time, builds a personalized memory base, and offers intuitive visualization and secure local storage to ensure privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14993v2">Multi-modal Generative AI: Multi-modal LLMs, Diffusions and the Unification</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 20 pages, 11 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03296v3">Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads. We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like VLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 49% (T4) and 37% (A10) higher throughput in long-output settings. APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07870v1">DocCHA: Towards LLM-Augmented Interactive Online diagnosis System</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Despite the impressive capabilities of Large Language Models (LLMs), existing Conversational Health Agents (CHAs) remain static and brittle, incapable of adaptive multi-turn reasoning, symptom clarification, or transparent decision-making. This hinders their real-world applicability in clinical diagnosis, where iterative and structured dialogue is essential. We propose DocCHA, a confidence-aware, modular framework that emulates clinical reasoning by decomposing the diagnostic process into three stages: (1) symptom elicitation, (2) history acquisition, and (3) causal graph construction. Each module uses interpretable confidence scores to guide adaptive questioning, prioritize informative clarifications, and refine weak reasoning links. Evaluated on two real-world Chinese consultation datasets (IMCS21, DX), DocCHA consistently outperforms strong prompting-based LLM baselines (GPT-3.5, GPT-4o, LLaMA-3), achieving up to 5.18 percent higher diagnostic accuracy and over 30 percent improvement in symptom recall, with only modest increase in dialogue turns. These results demonstrate the effectiveness of DocCHA in enabling structured, transparent, and efficient diagnostic conversations -- paving the way for trustworthy LLM-powered clinical assistants in multilingual and resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v2">The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v5">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02357v2">Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint. This work has been submitted to the Technical AI Governance Workshop at ICML 2025 for review
    </div>
    <details class="paper-abstract">
      Credible safety plans for advanced AI development require methods to verify agent behavior and detect potential control deficiencies early. A fundamental aspect is ensuring agents adhere to safety-critical principles, especially when these conflict with operational goals. This paper introduces a lightweight, interpretable benchmark to evaluate an LLM agent's ability to uphold a high-level safety principle when faced with conflicting task instructions. Our evaluation of six LLMs reveals two primary findings: (1) a quantifiable "cost of compliance" where safety constraints degrade task performance even when compliant solutions exist, and (2) an "illusion of compliance" where high adherence often masks task incompetence rather than principled choice. These findings provide initial evidence that while LLMs can be influenced by hierarchical directives, current approaches lack the consistency required for reliable safety governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01936v2">The Thin Line Between Comprehension and Persuasion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
    </details>
</div>
