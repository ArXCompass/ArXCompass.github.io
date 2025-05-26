# llm - 2025_05

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13418v1">Dementia Through Different Eyes: Explainable Modeling of Human and LLM Perceptions for Early Awareness</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Cognitive decline often surfaces in language years before diagnosis. It is frequently non-experts, such as those closest to the patient, who first sense a change and raise concern. As LLMs become integrated into daily communication and used over prolonged periods, it may even be an LLM that notices something is off. But what exactly do they notice--and should be noticing--when making that judgment? This paper investigates how dementia is perceived through language by non-experts. We presented transcribed picture descriptions to non-expert humans and LLMs, asking them to intuitively judge whether each text was produced by someone healthy or with dementia. We introduce an explainable method that uses LLMs to extract high-level, expert-guided features representing these picture descriptions, and use logistic regression to model human and LLM perceptions and compare with clinical diagnoses. Our analysis reveals that human perception of dementia is inconsistent and relies on a narrow, and sometimes misleading, set of cues. LLMs, by contrast, draw on a richer, more nuanced feature set that aligns more closely with clinical patterns. Still, both groups show a tendency toward false negatives, frequently overlooking dementia cases. Through our interpretable framework and the insights it provides, we hope to help non-experts better recognize the linguistic signs that matter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13416v1">Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice of LMO-based Optimizers for LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Recent developments in deep learning optimization have brought about radically new algorithms based on the Linear Minimization Oracle (LMO) framework, such as $\sf Muon$ and $\sf Scion$. After over a decade of $\sf Adam$'s dominance, these LMO-based methods are emerging as viable replacements, offering several practical advantages such as improved memory efficiency, better hyperparameter transferability, and most importantly, superior empirical performance on large-scale tasks, including LLM training. However, a significant gap remains between their practical use and our current theoretical understanding: prior analyses (1) overlook the layer-wise LMO application of these optimizers in practice, and (2) rely on an unrealistic smoothness assumption, leading to impractically small stepsizes. To address both, we propose a new LMO-based method called $\sf Gluon$, capturing prior theoretically analyzed methods as special cases, and introduce a new refined generalized smoothness model that captures the layer-wise geometry of neural networks, matches the layer-wise practical implementation of $\sf Muon$ and $\sf Scion$, and leads to convergence guarantees with strong practical predictive power. Unlike prior results, our theoretical stepsizes closely match the fine-tuned values reported by Pethick et al. (2025). Our experiments with NanoGPT and CNN confirm that our assumption holds along the optimization trajectory, ultimately closing the gap between theory and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13406v1">AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      A mathematical knowledge graph (KG) presents knowledge within the field of mathematics in a structured manner. Constructing a math KG using natural language is an essential but challenging task. There are two major limitations of existing works: first, they are constrained by corpus completeness, often discarding or manually supplementing incomplete knowledge; second, they typically fail to fully automate the integration of diverse knowledge sources. This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG capable of automatic updates. AutoMathKG regards mathematics as a vast directed graph composed of Definition, Theorem, and Problem entities, with their reference relationships as edges. It integrates knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing entities and relationships with large language models (LLMs) via in-context learning for data augmentation. To search for similar entities, MathVD, a vector database, is built through two designed embedding strategies using SBERT. To automatically update, two mechanisms are proposed. For knowledge completion mechanism, Math LLM is developed to interact with AutoMathKG, providing missing proofs or solutions. For knowledge fusion mechanism, MathVD is used to retrieve similar entities, and LLM is used to determine whether to merge with a candidate or add as a new entity. A wide range of experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13379v1">Thinkless: LLM Learns When to Think</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at https://github.com/VainF/Thinkless
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13376v1">Seeing, Saying, Solving: An LLM-to-TL Framework for Cooperative Robots</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Increased robot deployment, such as in warehousing, has revealed a need for seamless collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To address this challenge, we propose a novel, decentralized framework for robots to request and provide help. The framework begins with robots detecting conflicts using a Vision Language Model (VLM), then reasoning over whether help is needed. If so, it crafts and broadcasts a natural language (NL) help request using a Large Language Model (LLM). Potential helper robots reason over the request and offer help (if able), along with information about impact to their current tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar to guarantee syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot chooses a helper by reasoning over impact on the overall system. We evaluate our system via experiments considering different strategies for choosing a helper, and find that a requester robot can minimize overall time impact on the system by considering multiple help offers versus simple heuristics (e.g., selecting the nearest robot to help).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11380v2">From the New World of Word Embeddings: A Comparative Study of Small-World Lexico-Semantic Networks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Paper under review
    </div>
    <details class="paper-abstract">
      Lexico-semantic networks represent words as nodes and their semantic relatedness as edges. While such networks are traditionally constructed using embeddings from encoder-based models or static vectors, embeddings from decoder-only large language models (LLMs) remain underexplored. Unlike encoder models, LLMs are trained with a next-token prediction objective, which does not directly encode the meaning of the current token. In this paper, we construct lexico-semantic networks from the input embeddings of LLMs with varying parameter scales and conduct a comparative analysis of their global and local structures. Our results show that these networks exhibit small-world properties, characterized by high clustering and short path lengths. Moreover, larger LLMs yield more intricate networks with less small-world effects and longer paths, reflecting richer semantic structures and relations. We further validate our approach through analyses of common conceptual pairs, structured lexical relations derived from WordNet, and a cross-lingual semantic network for qualitative words.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13360v1">What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Building LLM-powered software requires developers to communicate their requirements through natural language, but developer prompts are frequently underspecified, failing to fully capture many user-important requirements. In this paper, we present an in-depth analysis of prompt underspecification, showing that while LLMs can often (41.1%) guess unspecified requirements by default, such behavior is less robust: Underspecified prompts are 2x more likely to regress over model or prompt changes, sometimes with accuracy drops by more than 20%. We then demonstrate that simply adding more requirements to a prompt does not reliably improve performance, due to LLMs' limited instruction-following capabilities and competing constraints, and standard prompt optimizers do not offer much help. To address this, we introduce novel requirements-aware prompt optimization mechanisms that can improve performance by 4.8% on average over baselines that naively specify everything in the prompt. Beyond prompt optimization, we envision that effectively managing prompt underspecification requires a broader process, including proactive requirements discovery, evaluation, and monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13348v1">Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08140v2">Lost in Transmission: When and Why LLMs Fail to Reason Globally</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4o, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13326v1">Thinking Short and Right Over Thinking Long: Serving LLM Reasoning Efficiently and Accurately</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Recent advances in test-time scaling suggest that Large Language Models (LLMs) can gain better capabilities by generating Chain-of-Thought reasoning (analogous to human thinking) to respond a given request, and meanwhile exploring more reasoning branches (i.e., generating multiple responses and ensembling them) can improve the final output quality. However, when incorporating the two scaling dimensions, we find that the system efficiency is dampened significantly for two reasons. Firstly, the time cost to generate the final output increases substantially as many reasoning branches would be trapped in the over-thinking dilemma, producing excessively long responses. Secondly, generating multiple reasoning branches for each request increases memory consumption, which is unsuitable for LLM serving since we can only batch a limited number of requests to process simultaneously. To address this, we present SART, a serving framework for efficient and accurate LLM reasoning. The essential idea is to manage the thinking to be short and right, rather than long. For one thing, we devise a redundant sampling with early stopping approach based on empirical observations and theoretic analysis, which increases the likelihood of obtaining short-thinking responses when sampling reasoning branches. For another, we propose to dynamically prune low-quality branches so that only right-thinking branches are maintained, reducing the memory consumption and allowing us to batch more requests. Experimental results demonstrate that SART not only improves the accuracy of LLM reasoning but also enhances the serving efficiency, outperforming existing methods by up to 28.2 times and on average 15.7 times in terms of efficiency when achieving the same level of accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13312v1">GUARD: Generation-time LLM Unlearning via Adaptive Restriction and Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in memorizing vast amounts of knowledge across diverse domains. However, the ability to selectively forget specific knowledge is critical for ensuring the safety and compliance of deployed models. Existing unlearning efforts typically fine-tune the model with resources such as forget data, retain data, and a calibration model. These additional gradient steps blur the decision boundary between forget and retain knowledge, making unlearning often at the expense of overall performance. To avoid the negative impact of fine-tuning, it would be better to unlearn solely at inference time by safely guarding the model against generating responses related to the forget target, without destroying the fluency of text generation. In this work, we propose Generation-time Unlearning via Adaptive Restriction and Detection (GUARD), a framework that enables dynamic unlearning during LLM generation. Specifically, we first employ a prompt classifier to detect unlearning targets and extract the corresponding forbidden token. We then dynamically penalize and filter candidate tokens during generation using a combination of token matching and semantic matching, effectively preventing the model from leaking the forgotten content. Experimental results on copyright content unlearning tasks over the Harry Potter dataset and the MUSE benchmark, as well as entity unlearning tasks on the TOFU dataset, demonstrate that GUARD achieves strong forget quality across various tasks while causing almost no degradation to the LLM's general capabilities, striking an excellent trade-off between forgetting and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17140v2">AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have enabled LLM-based agents to directly interact with application user interfaces (UIs), enhancing agents' performance in complex tasks. However, these agents often suffer from high latency and low reliability due to the extensive sequential UI interactions. To address this issue, we propose AXIS, a novel LLM-based agents framework that prioritize actions through application programming interfaces (APIs) over UI actions. This framework also facilitates the creation and expansion of APIs through automated exploration of applications. Our experiments on Microsoft Word demonstrate that AXIS reduces task completion time by 65%-70% and cognitive workload by 38%-53%, while maintaining accuracy of 97%-98% compared to humans. Our work contributes to a new human-agent-computer interaction (HACI) framework and explores a fresh UI design principle for application providers to turn applications into agents in the era of LLMs, paving the way towards an agent-centric operating system (Agent OS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09493v2">Hacking, The Lazy Way: LLM Augmented Pentesting</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 This work has been submitted to the IEEE for possible publication. Nisha P. Shetty has been added as an author as the new version includes work under her supervision, enhancing the research. Significant changes have been made in the methodology, survey, and introduction sections
    </div>
    <details class="paper-abstract">
      In our research, we introduce a new concept called "LLM Augmented Pentesting" demonstrated with a tool named "Pentest Copilot," that revolutionizes the field of ethical hacking by integrating Large Language Models (LLMs) into penetration testing workflows, leveraging the advanced GPT-4-turbo model. Our approach focuses on overcoming the traditional resistance to automation in penetration testing by employing LLMs to automate specific sub-tasks while ensuring a comprehensive understanding of the overall testing process. Pentest Copilot showcases remarkable proficiency in tasks such as utilizing testing tools, interpreting outputs, and suggesting follow-up actions, efficiently bridging the gap between automated systems and human expertise. By integrating a "chain of thought" mechanism, Pentest Copilot optimizes token usage and enhances decision-making processes, leading to more accurate and context-aware outputs. Additionally, our implementation of Retrieval-Augmented Generation (RAG) minimizes hallucinations and ensures the tool remains aligned with the latest cybersecurity techniques and knowledge. We also highlight a unique infrastructure system that supports in-browser penetration testing, providing a robust platform for cybersecurity professionals. Our findings demonstrate that LLM Augmented Pentesting can not only significantly enhance task completion rates in penetration testing but also effectively addresses real-world challenges, marking a substantial advancement in the cybersecurity domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16707v3">Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The integration of experimental technologies with large language models (LLMs) is transforming scientific research. It positions AI as a versatile research assistant rather than a mere problem-solving tool. In the field of power systems, however, managing simulations -- one of the essential experimental technologies -- remains a challenge for LLMs due to their limited domain-specific knowledge, restricted reasoning capabilities, and imprecise handling of simulation parameters. To address these limitations, this paper proposes a feedback-driven, multi-agent framework. It incorporates three proposed modules: an enhanced retrieval-augmented generation (RAG) module, an improved reasoning module, and a dynamic environmental acting module with an error-feedback mechanism. Validated on 69 diverse tasks from Daline and MATPOWER, this framework achieves success rates of 93.13% and 96.85%, respectively. It significantly outperforms ChatGPT 4o, o1-preview, and the fine-tuned GPT-4o, which all achieved a success rate lower than 30% on complex tasks. Additionally, the proposed framework also supports rapid, cost-effective task execution, completing each simulation in approximately 30 seconds at an average cost of 0.014 USD for tokens. Overall, this adaptable framework lays a foundation for developing intelligent LLM-based assistants for human researchers, facilitating power system research and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13263v1">Are requirements really all you need? A case study of LLM-driven configuration code generation for automotive simulations</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are taking many industries by storm. They possess impressive reasoning capabilities and are capable of handling complex problems, as shown by their steadily improving scores on coding and mathematical benchmarks. However, are the models currently available truly capable of addressing real-world challenges, such as those found in the automotive industry? How well can they understand high-level, abstract instructions? Can they translate these instructions directly into functional code, or do they still need help and supervision? In this work, we put one of the current state-of-the-art models to the test. We evaluate its performance in the task of translating abstract requirements, extracted from automotive standards and documents, into configuration code for CARLA simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13246v1">Agentic Publications: An LLM-Driven Framework for Interactive Scientific Publishing, Supplementing Traditional Papers with AI-Powered Knowledge Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The exponential growth of scientific literature presents significant challenges for researchers navigating the complex knowledge landscape. We propose "Agentic Publications", a novel LLM-driven framework complementing traditional publishing by transforming papers into interactive knowledge systems. Our architecture integrates structured data with unstructured content through retrieval-augmented generation and multi-agent verification. The framework offers interfaces for both humans and machines, combining narrative explanations with machine-readable outputs while addressing ethical considerations through automated validation and transparent governance. Key features include continuous knowledge updates, automatic integration of new findings, and customizable detail levels. Our proof-of-concept demonstrates multilingual interaction, API accessibility, and structured knowledge representation through vector databases, knowledge graphs, and verification agents. This approach enhances scientific communication across disciplines, improving efficiency and collaboration while preserving traditional publishing pathways, particularly valuable for interdisciplinary fields where knowledge integration remains challenging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11266v5">VersaTune: An Efficient Data Composition Framework for Training Multi-Capability LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As demonstrated by the proprietary Large Language Models (LLMs) such as GPT and Claude series, LLMs have the potential to achieve remarkable proficiency across a wide range of domains, including law, medicine, finance, science, code, etc., all within a single model. These capabilities are further augmented during the Supervised Fine-Tuning (SFT) phase. Despite their potential, existing work mainly focuses on domain-specific enhancements during fine-tuning, the challenge of which lies in catastrophic forgetting of knowledge across other domains. In this study, we introduce **VersaTune**, a novel data composition framework designed for enhancing LLMs' overall multi-domain capabilities during training. We begin with detecting the distribution of domain-specific knowledge within the base model, followed by the training data composition that aligns with the model's existing knowledge distribution. During the subsequent training process, domain weights are dynamically adjusted based on their learnable potential and forgetting degree. Experimental results indicate that VersaTune is effective in multi-domain fostering, with an improvement of 35.21\% in the overall multi-ability performances compared to uniform domain weights. Furthermore, we find that Qwen-2.5-32B + VersaTune even surpasses frontier models, including GPT-4o, Claude3.5-Sonnet and DeepSeek-V3 by 0.86\%, 4.76\% and 4.60\%. Additionally, in scenarios where flexible expansion of a specific domain is required, VersaTune reduces the performance degradation in other domains by 38.77\%, while preserving the training efficacy of the target domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13195v1">Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly integrated into real-world decision-making systems, understanding their behavioural vulnerabilities remains a critical challenge for AI safety and alignment. While existing evaluation metrics focus primarily on reasoning accuracy or factual correctness, they often overlook whether LLMs are robust to adversarial manipulation or capable of using adaptive strategy in dynamic environments. This paper introduces an adversarial evaluation framework designed to systematically stress-test the decision-making processes of LLMs under interactive and adversarial conditions. Drawing on methodologies from cognitive psychology and game theory, our framework probes how models respond in two canonical tasks: the two-armed bandit task and the Multi-Round Trust Task. These tasks capture key aspects of exploration-exploitation trade-offs, social cooperation, and strategic flexibility. We apply this framework to several state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3, revealing model-specific susceptibilities to manipulation and rigidity in strategy adaptation. Our findings highlight distinct behavioral patterns across models and emphasize the importance of adaptability and fairness recognition for trustworthy AI deployment. Rather than offering a performance benchmark, this work proposes a methodology for diagnosing decision-making weaknesses in LLM-based agents, providing actionable insights for alignment and safety research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13175v1">Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The emerging paradigm of leveraging pretrained large language models (LLMs) for time series forecasting has predominantly employed linguistic-temporal modality alignment strategies through token-level or layer-wise feature mapping. However, these approaches fundamentally neglect a critical insight: the core competency of LLMs resides not merely in processing localized token features but in their inherent capacity to model holistic sequence structures. This paper posits that effective cross-modal alignment necessitates structural consistency at the sequence level. We propose the Structure-Guided Cross-Modal Alignment (SGCMA), a framework that fully exploits and aligns the state-transition graph structures shared by time-series and linguistic data as sequential modalities, thereby endowing time series with language-like properties and delivering stronger generalization after modality alignment. SGCMA consists of two key components, namely Structure Alignment and Semantic Alignment. In Structure Alignment, a state transition matrix is learned from text data through Hidden Markov Models (HMMs), and a shallow transformer-based Maximum Entropy Markov Model (MEMM) receives the hot-start transition matrix and annotates each temporal patch into state probability, ensuring that the temporal representation sequence inherits language-like sequential dynamics. In Semantic Alignment, cross-attention is applied between temporal patches and the top-k tokens within each state, and the ultimate temporal embeddings are derived by the expected value of these embeddings using a weighted average based on state probabilities. Experiments on multiple benchmarks demonstrate that SGCMA achieves state-of-the-art performance, offering a novel approach to cross-modal alignment in time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13173v1">A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across diverse tasks and languages. In this study, we focus on natural language understanding in three classical languages -- Sanskrit, Ancient Greek and Latin -- to investigate the factors affecting cross-lingual zero-shot generalization. First, we explore named entity recognition and machine translation into English. While LLMs perform equal to or better than fine-tuned baselines on out-of-domain data, smaller models often struggle, especially with niche or abstract entity types. In addition, we concentrate on Sanskrit by presenting a factoid question-answering (QA) dataset and show that incorporating context via retrieval-augmented generation approach significantly boosts performance. In contrast, we observe pronounced performance drops for smaller LLMs across these QA tasks. These results suggest model scale as an important factor influencing cross-lingual generalization. Assuming that models used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical languages, our findings provide insights into how LLMs may generalize on these languages and their consequent utility in classical studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02613v2">ACCO: Accumulate While You Communicate for Communication-Overlapped Sharded LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Training LLMs relies on distributed implementations using multiple GPUs to compute gradients in parallel with sharded optimizers. However, synchronizing gradients in data parallel setups introduces communication overhead that grows with the number of workers, limiting parallelization efficiency. Local optimization algorithms reduce communications but incur high memory costs as they prevent optimizer state sharding, hindering scalability. To address this, we propose \textbf{AC}cumulate while \textbf{CO}mmunicate (\acco), a memory-efficient optimization algorithm for distributed LLM training. By synchronizing delayed gradients while computing new ones, \acco~reduces GPU idle time and supports heterogeneous hardware. To mitigate the convergence issues caused by delayed updates, we introduce a novel technique ensuring training dynamics align with standard distributed optimization. Compared to ZeRO-1, our approach is significantly faster and scales effectively across heterogeneous hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13691v2">Is This Collection Worth My LLM's Time? Automatically Measuring Information Potential in Text Corpora</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) converge towards similar capabilities, the key to advancing their performance lies in identifying and incorporating valuable new information sources. However, evaluating which text collections are worth the substantial investment required for digitization, preprocessing, and integration into LLM systems remains a significant challenge. We present a novel approach to this challenge: an automated pipeline that evaluates the potential information gain from text collections without requiring model training or fine-tuning. Our method generates multiple choice questions (MCQs) from texts and measures an LLM's performance both with and without access to the source material. The performance gap between these conditions serves as a proxy for the collection's information potential. We validate our approach using five strategically selected datasets: EPFL PhD manuscripts, a private collection of Venetian historical records, two sets of Wikipedia articles on related topics, and a synthetic baseline dataset. Our results demonstrate that this method effectively identifies collections containing valuable novel information, providing a practical tool for prioritizing data acquisition and integration efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07610v2">Concept-Level Explainability for Auditing & Steering LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 9 pages, 7 figures, Submission to Neurips 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11525v2">Beyond Single-Task: Robust Multi-Task Length Generalization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Length generalization, the ability to solve problems longer than those seen during training, remains a critical challenge for large language models (LLMs). Previous work modifies positional encodings (PEs) and data formats to improve length generalization on specific symbolic tasks such as addition and sorting. However, these approaches are fundamentally limited to special tasks, often degrading general language performance. Furthermore, they are typically evaluated on small transformers trained from scratch on single tasks and can cause performance drop when applied during post-training stage of practical LLMs with general capabilities. Hu et al., (2024) proposed Rule-Following Fine-Tuning (RFFT) to improve length generalization in the post-training stage of LLMs. Despite its compatibility with practical models and strong performance, RFFT is proposed for single tasks too, requiring re-training for each individual task with extensive examples. In this paper, we study length generalization in multi-task settings and propose Meta Rule-Following Fine-Tuning (Meta-RFFT), the first framework enabling robust cross-task length generalization. As our first contribution, we construct a large length generalization dataset containing 86 tasks spanning code execution, number processing, symbolic and logical reasoning tasks, beyond the common addition or multiplication tasks. Secondly, we show that cross-task length generalization is possible with Meta-RFFT. After training on a large number of tasks and instances, the models achieve remarkable length generalization ability on unseen tasks with minimal fine-tuning or one-shot prompting. For example, after fine-tuning on 1 to 5 digit addition, our 32B model achieves 95% accuracy on 30 digit addition, significantly outperforming the state-of-the-art reasoning models (DeepSeek-R1-671B: 72%), despite never seeing this task during RF-pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13109v1">FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13098v1">LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Peer reviewed publication at ESWC 2025 Resources Track
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13028v1">Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13026v1">Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13004v1">EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Existing code generation benchmarks primarily evaluate functional correctness, with limited focus on code efficiency and often restricted to a single language like Python. To address this gap, we introduce EffiBench-X, the first multi-language benchmark designed to measure the efficiency of LLM-generated code. EffiBench-X supports Python, C++, Java, JavaScript, Ruby, and Golang. It comprises competitive programming tasks with human-expert solutions as efficiency baselines. Evaluating state-of-the-art LLMs on EffiBench-X reveals that while models generate functionally correct code, they consistently underperform human experts in efficiency. Even the most efficient LLM-generated solutions (Qwen3-32B) achieve only around \textbf{62\%} of human efficiency on average, with significant language-specific variations. LLMs show better efficiency in Python, Ruby, and JavaScript than in Java, C++, and Golang. For instance, DeepSeek-R1's Python code is significantly more efficient than its Java code. These results highlight the critical need for research into LLM optimization techniques to improve code efficiency across diverse languages. The dataset and evaluation infrastructure are submitted and available at https://github.com/EffiBench/EffiBench-X.git and https://huggingface.co/datasets/EffiBench/effibench-x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12991v1">Personalized Fine-Tuning with Controllable Synthetic Speech from LLM-Generated Transcripts for Dysarthric Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Accepted at Interspeech 2025
    </div>
    <details class="paper-abstract">
      In this work, we present our submission to the Speech Accessibility Project challenge for dysarthric speech recognition. We integrate parameter-efficient fine-tuning with latent audio representations to improve an encoder-decoder ASR system. Synthetic training data is generated by fine-tuning Parler-TTS to mimic dysarthric speech, using LLM-generated prompts for corpus-consistent target transcripts. Personalization with x-vectors consistently reduces word error rates (WERs) over non-personalized fine-tuning. AdaLoRA adapters outperform full fine-tuning and standard low-rank adaptation, achieving relative WER reductions of ~23% and ~22%, respectively. Further improvements (~5% WER reduction) come from incorporating wav2vec 2.0-based audio representations. Training with synthetic dysarthric speech yields up to ~7% relative WER improvement over personalized fine-tuning alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12981v1">From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08946v2">Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 43 pages, 6 figures, including the latest works published in 2024-2025
    </div>
    <details class="paper-abstract">
      Explainable AI (XAI) refers to techniques that provide human-understandable insights into the workings of AI models. Recently, the focus of XAI is being extended toward explaining Large Language Models (LLMs). This extension calls for a significant transformation in the XAI methodologies for two reasons. First, many existing XAI methods cannot be directly applied to LLMs due to their complexity and advanced capabilities. Second, as LLMs are increasingly deployed in diverse applications, the role of XAI shifts from merely opening the ``black box'' to actively enhancing the productivity and applicability of LLMs in real-world settings. Meanwhile, the conversation and generation abilities of LLMs can reciprocally enhance XAI. Therefore, in this paper, we introduce Usable XAI in the context of LLMs by analyzing (1) how XAI can explain and improve LLM-based AI systems and (2) how XAI techniques can be improved by using LLMs. We introduce 10 strategies, introducing the key techniques for each and discussing their associated challenges. We also provide case studies to demonstrate how to obtain and leverage explanations. The code used in this paper can be found at: https://github.com/JacksonWuxs/UsableXAI_LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12570v1">Batched Self-Consistency Improves LLM Relevance Assessment and Ranking</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Given some information need, Large Language Models (LLMs) are increasingly used for candidate text relevance assessment, typically using a one-by-one pointwise (PW) strategy where each LLM call evaluates one candidate at a time. Meanwhile, it has been shown that LLM performance can be improved through self-consistency: prompting the LLM to do the same task multiple times (possibly in perturbed ways) and then aggregating the responses. To take advantage of self-consistency, we hypothesize that batched PW strategies, where multiple passages are judged in one LLM call, are better suited than one-by-one PW methods since a larger input context can induce more diverse LLM sampling across self-consistency calls. We first propose several candidate batching strategies to create prompt diversity across self-consistency calls through subset reselection and permutation. We then test our batched PW methods on relevance assessment and ranking tasks against one-by-one PW and listwise LLM ranking baselines with and without self-consistency, using three passage retrieval datasets and GPT-4o, Claude Sonnet 3, and Amazon Nova Pro. We find that batched PW methods outperform all baselines, and show that batching can greatly amplify the positive effects of self-consistency. For instance, on our legal search dataset, GPT-4o one-by-one PW ranking NDCG@10 improves only from 44.9% to 46.8% without self-consistency vs. with 15 self consistency calls, while batched PW ranking improves from 43.8% to 51.3%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12543v1">Disambiguation in Conversational Question Answering in the Era of LLM: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable language systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12501v1">ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 36 pages, 10 figures, 19 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at rapid generation of text and multimodal content, yet they falter on transaction-style planning that demands ACID-like guarantees and real-time disruption recovery. We present Adaptive LLM Agent System (ALAS), a framework that tackles four fundamental LLM deficits: (i) absence of self-verification, (ii) context erosion, (iii) next-token myopia, and (iv) lack of persistent state. ALAS decomposes each plan into role-specialized agents, equips them with automatic state tracking, and coordinates them through a lightweight protocol. When disruptions arise, agents apply history-aware local compensation, avoiding costly global replanning and containing cascade effects. On real-world, large-scale job-shop scheduling benchmarks, ALAS sets new best results for static sequential planning and excels in dynamic reactive scenarios with unexpected disruptions. These gains show that principled modularization plus targeted compensation can unlock scalable and resilient planning with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12500v1">MARGE: Improving Math Reasoning for LLMs with Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 To appear at ICML 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong potential in mathematical reasoning, yet their effectiveness is often limited by a shortage of high-quality queries. This limitation necessitates scaling up computational responses through self-generated data, yet current methods struggle due to spurious correlated data caused by ineffective exploration across all reasoning stages. To address such challenge, we introduce \textbf{MARGE}: Improving \textbf{Ma}th \textbf{R}easoning with \textbf{G}uided \textbf{E}xploration, a novel method to address this issue and enhance mathematical reasoning through hit-guided exploration. MARGE systematically explores intermediate reasoning states derived from self-generated solutions, enabling adequate exploration and improved credit assignment throughout the reasoning process. Through extensive experiments across multiple backbone models and benchmarks, we demonstrate that MARGE significantly improves reasoning capabilities without requiring external annotations or training additional value models. Notably, MARGE improves both single-shot accuracy and exploration diversity, mitigating a common trade-off in alignment methods. These results demonstrate MARGE's effectiveness in enhancing mathematical reasoning capabilities and unlocking the potential of scaling self-generated training data. Our code and models are available at \href{https://github.com/georgao35/MARGE}{this link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16638v3">LLMScan: Causal Scan for LLM Misbehavior Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution. LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12495v1">KG-QAGen: A Knowledge-Graph-Based Framework for Systematic Question Generation and Long-Context LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      The increasing context length of modern language models has created a need for evaluating their ability to retrieve and process information across extensive documents. While existing benchmarks test long-context capabilities, they often lack a structured way to systematically vary question complexity. We introduce KG-QAGen (Knowledge-Graph-based Question-Answer Generation), a framework that (1) extracts QA pairs at multiple complexity levels (2) by leveraging structured representations of financial agreements (3) along three key dimensions -- multi-hop retrieval, set operations, and answer plurality -- enabling fine-grained assessment of model performance across controlled difficulty levels. Using this framework, we construct a dataset of 20,139 QA pairs (the largest number among the long-context benchmarks) and open-source a part of it. We evaluate 13 proprietary and open-source LLMs and observe that even the best-performing models are struggling with set-based comparisons and multi-hop logical inference. Our analysis reveals systematic failure modes tied to semantic misinterpretation and inability to handle implicit relations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12452v1">Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 We commit to fully open-source our patent dataset
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models with better understanding do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12442v1">IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12439v1">Learning to Play Like Humans: A Framework for LLM Adaptation in Interactive Fiction Games</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Interactive Fiction games (IF games) are where players interact through natural language commands. While recent advances in Artificial Intelligence agents have reignited interest in IF games as a domain for studying decision-making, existing approaches prioritize task-specific performance metrics over human-like comprehension of narrative context and gameplay logic. This work presents a cognitively inspired framework that guides Large Language Models (LLMs) to learn and play IF games systematically. Our proposed **L**earning to **P**lay **L**ike **H**umans (LPLH) framework integrates three key components: (1) structured map building to capture spatial and narrative relationships, (2) action learning to identify context-appropriate commands, and (3) feedback-driven experience analysis to refine decision-making over time. By aligning LLMs-based agents' behavior with narrative intent and commonsense constraints, LPLH moves beyond purely exploratory strategies to deliver more interpretable, human-like performance. Crucially, this approach draws on cognitive science principles to more closely simulate how human players read, interpret, and respond within narrative worlds. As a result, LPLH reframes the IF games challenge as a learning problem for LLMs-based agents, offering a new path toward robust, context-aware gameplay in complex text-based environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12424v1">EvoGPT: Enhancing Test Suite Robustness via LLM-Based Generation and Genetic Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as promising tools for automated unit test generation. We introduce a hybrid framework called EvoGPT that integrates LLM-based test generation with evolutionary search techniques to create diverse, fault-revealing unit tests. Unit tests are initially generated with diverse temperature sampling to maximize behavioral and test suite diversity, followed by a generation-repair loop and coverage-guided assertion enhancement. The resulting test suites are evolved using genetic algorithms, guided by a fitness function prioritizing mutation score over traditional coverage metrics. This design emphasizes the primary objective of unit testing-fault detection. Evaluated on multiple open-source Java projects, EvoGPT achieves an average improvement of 10% in both code coverage and mutation score compared to LLMs and traditional search-based software testing baselines. These results demonstrate that combining LLM-driven diversity, targeted repair, and evolutionary optimization produces more effective and resilient test suites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05812v3">Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 Ongoing work. First released on April 8, 2025
    </div>
    <details class="paper-abstract">
      Existing methods to enhance the reasoning capability of large language models predominantly rely on supervised fine-tuning (SFT) followed by reinforcement learning (RL) on reasoning-specific data. These approaches critically depend on external supervisions--such as labeled reasoning traces, verified golden answers, or pre-trained reward models. In this work, we propose Entropy Minimized Policy Optimization (\ours), which makes an early attempt at fully unsupervised LLM reasoning incentivization. By continuously minimizing the predictive entropy of LLMs on unlabeled questions in a latent semantic space, \ours achieves competitive performance compared to supervised counterparts on both mathematical and free-form natural reasoning tasks. Specifically, without any supervised signals, \ours boosts the accuracy of Qwen2.5-Math-7B Base from 30.7\% to 48.1\% on mathematical benchmarks and improves the accuracy of Qwen2.5-7B Base from 32.1\% to 50.1\% on MMLU-Pro. Primary experiments and analysis are also provided to interpret the effectiveness of \ours. Code is available at https://github.com/QingyangZhang/EMPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12396v1">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02707v4">LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often produce errors, including factual inaccuracies, biases, and reasoning failures, collectively referred to as "hallucinations". Recent studies have demonstrated that LLMs' internal states encode information regarding the truthfulness of their outputs, and that this information can be utilized to detect errors. In this work, we show that the internal representations of LLMs encode much more information about truthfulness than previously recognized. We first discover that the truthfulness information is concentrated in specific tokens, and leveraging this property significantly enhances error detection performance. Yet, we show that such error detectors fail to generalize across datasets, implying that -- contrary to prior claims -- truthfulness encoding is not universal but rather multifaceted. Next, we show that internal representations can also be used for predicting the types of errors the model is likely to make, facilitating the development of tailored mitigation strategies. Lastly, we reveal a discrepancy between LLMs' internal encoding and external behavior: they may encode the correct answer, yet consistently generate an incorrect one. Taken together, these insights deepen our understanding of LLM errors from the model's internal perspective, which can guide future research on enhancing error analysis and mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12349v1">Wisdom from Diversity: Bias Mitigation Through Hybrid Human-LLM Crowds</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 Accepted for publication in the Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
    </div>
    <details class="paper-abstract">
      Despite their performance, large language models (LLMs) can inadvertently perpetuate biases found in the data they are trained on. By analyzing LLM responses to bias-eliciting headlines, we find that these models often mirror human biases. To address this, we explore crowd-based strategies for mitigating bias through response aggregation. We first demonstrate that simply averaging responses from multiple LLMs, intended to leverage the "wisdom of the crowd", can exacerbate existing biases due to the limited diversity within LLM crowds. In contrast, we show that locally weighted aggregation methods more effectively leverage the wisdom of the LLM crowd, achieving both bias mitigation and improved accuracy. Finally, recognizing the complementary strengths of LLMs (accuracy) and humans (diversity), we demonstrate that hybrid crowds containing both significantly enhance performance and further reduce biases across ethnic and gender-related contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12348v1">Reasoning-CV: Fine-tuning Powerful Reasoning LLMs for Knowledge-Assisted Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Claim verification is essential in combating misinformation, and large language models (LLMs) have recently emerged in this area as powerful tools for assessing the veracity of claims using external knowledge. Existing LLM-based methods for claim verification typically adopt a Decompose-Then-Verify paradigm, which involves decomposing complex claims into several independent sub-claims and verifying each sub-claim separately. However, this paradigm often introduces errors during the claim decomposition process. To mitigate these errors, we propose to develop the Chain-of-Thought (CoT)-Verify paradigm, which leverages LLM reasoning methods to generate CoT-verification paths for the original complex claim without requiring decompositions into sub-claims and separate verification stages. The CoT-Verify paradigm allows us to propose a natural fine-tuning method called Reasoning-CV to enhance the verification capabilities in LLMs. Reasoning-CV includes a supervised fine-tuning (SFT) stage and a self-improvement direct preference optimization (DPO) stage. Utilizing only an 8B pre-trained LLM, Reasoning-CV demonstrates superior knowledge-assisted claim verification performances compared to existing Decompose-Then-Verify methods, as well as powerful black-box LLMs such as GPT-4o+CoT and o1-preview. Our code is available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12331v1">OSS-Bench: Benchmark Generator for Coding LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      In light of the rapid adoption of AI coding assistants, LLM-assisted development has become increasingly prevalent, creating an urgent need for robust evaluation of generated code quality. Existing benchmarks often require extensive manual effort to create static datasets, rely on indirect or insufficiently challenging tasks, depend on non-scalable ground truth, or neglect critical low-level security evaluations, particularly memory-safety issues. In this work, we introduce OSS-Bench, a benchmark generator that automatically constructs large-scale, live evaluation tasks from real-world open-source software. OSS-Bench replaces functions with LLM-generated code and evaluates them using three natural metrics: compilability, functional correctness, and memory safety, leveraging robust signals like compilation failures, test-suite violations, and sanitizer alerts as ground truth. In our evaluation, the benchmark, instantiated as OSS-Bench(php) and OSS-Bench(sql), profiles 17 diverse LLMs, revealing insights such as intra-family behavioral patterns and inconsistencies between model size and performance. Our results demonstrate that OSS-Bench mitigates overfitting by leveraging the evolving complexity of OSS and highlights LLMs' limited understanding of low-level code security via extended fuzzing experiments. Overall, OSS-Bench offers a practical and scalable framework for benchmarking the real-world coding capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12328v1">LLMSR@XLLM25: An Empirical Study of LLM for Structural Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      We present Team asdfo123's submission to the LLMSR@XLLM25 shared task, which evaluates large language models on producing fine-grained, controllable, and interpretable reasoning processes. Systems must extract all problem conditions, decompose a chain of thought into statement-evidence pairs, and verify the logical validity of each pair. Leveraging only the off-the-shelf Meta-Llama-3-8B-Instruct, we craft a concise few-shot, multi-turn prompt that first enumerates all conditions and then guides the model to label, cite, and adjudicate every reasoning step. A lightweight post-processor based on regular expressions normalises spans and enforces the official JSON schema. Without fine-tuning, external retrieval, or ensembling, our method ranks 5th overall, achieving macro F1 scores on par with substantially more complex and resource-consuming pipelines. We conclude by analysing the strengths and limitations of our approach and outlining directions for future research in structural reasoning with LLMs. Our code is available at https://github.com/asdfo123/LLMSR-asdfo123.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06207v3">Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 18 pages, accepted at the ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become essential for offensive language detection, yet their ability to handle annotation disagreement remains underexplored. Disagreement samples, which arise from subjective interpretations, pose a unique challenge due to their ambiguous nature. Understanding how LLMs process these cases, particularly their confidence levels, can offer insight into their alignment with human annotators. This study systematically evaluates the performance of multiple LLMs in detecting offensive language at varying levels of annotation agreement. We analyze binary classification accuracy, examine the relationship between model confidence and human disagreement, and explore how disagreement samples influence model decision-making during few-shot learning and instruction fine-tuning. Our findings reveal that LLMs struggle with low-agreement samples, often exhibiting overconfidence in these ambiguous cases. However, utilizing disagreement samples in training improves both detection accuracy and model alignment with human judgment. These insights provide a foundation for enhancing LLM-based offensive language detection in real-world moderation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12313v1">ExpertSteer: Intervening in LLMs through Expert Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable capabilities across various tasks, yet guiding them to follow desired behaviours during inference remains a significant challenge. Activation steering offers a promising method to control the generation process of LLMs by modifying their internal activations. However, existing methods commonly intervene in the model's behaviour using steering vectors generated by the model itself, which constrains their effectiveness to that specific model and excludes the possibility of leveraging powerful external expert models for steering. To address these limitations, we propose ExpertSteer, a novel approach that leverages arbitrary specialized expert models to generate steering vectors, enabling intervention in any LLMs. ExpertSteer transfers the knowledge from an expert model to a target LLM through a cohesive four-step process: first aligning representation dimensions with auto-encoders to enable cross-model transfer, then identifying intervention layer pairs based on mutual information analysis, next generating steering vectors from the expert model using Recursive Feature Machines, and finally applying these vectors on the identified layers during inference to selectively guide the target LLM without updating model parameters. We conduct comprehensive experiments using three LLMs on 15 popular benchmarks across four distinct domains. Experiments demonstrate that ExpertSteer significantly outperforms established baselines across diverse tasks at minimal cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12301v1">Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 19 pages, 3 tables, 3 figures
    </div>
    <details class="paper-abstract">
      LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10425v2">Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex tasks thanks to advances in reasoning abilities. However, existing methods overlook the trade-off between reasoning effectiveness and computational efficiency, often encouraging unnecessarily long reasoning chains and wasting tokens. To address this, we propose Learning to Think (L2T), an information-theoretic reinforcement fine-tuning framework for LLMs to make the models achieve optimal reasoning with fewer tokens. Specifically, L2T treats each query-response interaction as a hierarchical session of multiple episodes and proposes a universal dense process reward, i.e., quantifies the episode-wise information gain in parameters, requiring no extra annotations or task-specific evaluators. We propose a method to quickly estimate this reward based on PAC-Bayes bounds and the Fisher information matrix. Theoretical analyses show that it significantly reduces computational complexity with high estimation accuracy. By immediately rewarding each episode's contribution and penalizing excessive updates, L2T optimizes the model via reinforcement learning to maximize the use of each episode and achieve effective updates. Empirical results on various reasoning benchmarks and base models demonstrate the advantage of L2T across different tasks, boosting both reasoning effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00735v3">`Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12273v1">LLM-Based Evaluation of Low-Resource Machine Translation: A Reference-less Dialect Guided Approach with a Refined Sylheti-English Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Evaluating machine translation (MT) for low-resource languages poses a persistent challenge, primarily due to the limited availability of high quality reference translations. This issue is further exacerbated in languages with multiple dialects, where linguistic diversity and data scarcity hinder robust evaluation. Large Language Models (LLMs) present a promising solution through reference-free evaluation techniques; however, their effectiveness diminishes in the absence of dialect-specific context and tailored guidance. In this work, we propose a comprehensive framework that enhances LLM-based MT evaluation using a dialect guided approach. We extend the ONUBAD dataset by incorporating Sylheti-English sentence pairs, corresponding machine translations, and Direct Assessment (DA) scores annotated by native speakers. To address the vocabulary gap, we augment the tokenizer vocabulary with dialect-specific terms. We further introduce a regression head to enable scalar score prediction and design a dialect-guided (DG) prompting strategy. Our evaluation across multiple LLMs shows that the proposed pipeline consistently outperforms existing methods, achieving the highest gain of +0.1083 in Spearman correlation, along with improvements across other evaluation settings. The dataset and the code are available at https://github.com/180041123-Atiq/MTEonLowResourceLanguage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12260v1">LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12259v1">Teach2Eval: An Indirect Evaluation Method for LLM by Judging How It Teaches</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has outpaced the development of effective evaluation methods. Traditional benchmarks rely on task-specific metrics and static datasets, which often suffer from fairness issues, limited scalability, and contamination risks. In this paper, we introduce Teach2Eval, an indirect evaluation framework inspired by the Feynman Technique. Instead of directly testing LLMs on predefined tasks, our method evaluates a model's multiple abilities to teach weaker student models to perform tasks effectively. By converting open-ended tasks into standardized multiple-choice questions (MCQs) through teacher-generated feedback, Teach2Eval enables scalable, automated, and multi-dimensional assessment. Our approach not only avoids data leakage and memorization but also captures a broad range of cognitive abilities that are orthogonal to current benchmarks. Experimental results across 26 leading LLMs show strong alignment with existing human and model-based dynamic rankings, while offering additional interpretability for training guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12257v1">LLM Context Conditioning and PWP Prompting for Multimodal Validation of Chemical Formulas</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Identifying subtle technical errors within complex scientific and technical documents, especially those requiring multimodal interpretation (e.g., formulas in images), presents a significant hurdle for Large Language Models (LLMs) whose inherent error-correction tendencies can mask inaccuracies. This exploratory proof-of-concept (PoC) study investigates structured LLM context conditioning, informed by Persistent Workflow Prompting (PWP) principles, as a methodological strategy to modulate this LLM behavior at inference time. The approach is designed to enhance the reliability of readily available, general-purpose LLMs (specifically Gemini 2.5 Pro and ChatGPT Plus o3) for precise validation tasks, crucially relying only on their standard chat interfaces without API access or model modifications. To explore this methodology, we focused on validating chemical formulas within a single, complex test paper with known textual and image-based errors. Several prompting strategies were evaluated: while basic prompts proved unreliable, an approach adapting PWP structures to rigorously condition the LLM's analytical mindset appeared to improve textual error identification with both models. Notably, this method also guided Gemini 2.5 Pro to repeatedly identify a subtle image-based formula error previously overlooked during manual review, a task where ChatGPT Plus o3 failed in our tests. These preliminary findings highlight specific LLM operational modes that impede detail-oriented validation and suggest that PWP-informed context conditioning offers a promising and highly accessible technique for developing more robust LLM-driven analytical workflows, particularly for tasks requiring meticulous error detection in scientific and technical documents. Extensive validation beyond this limited PoC is necessary to ascertain broader applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01711v4">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 accepted to ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12238v1">PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      The memorization of sensitive and personally identifiable information (PII) by large language models (LLMs) poses growing privacy risks as models scale and are increasingly deployed in real-world applications. Existing efforts to study sensitive and PII data memorization and develop mitigation strategies are hampered by the absence of comprehensive, realistic, and ethically sourced datasets reflecting the diversity of sensitive information found on the web. We introduce PANORAMA - Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis, a large-scale synthetic corpus of 384,789 samples derived from 9,674 synthetic profiles designed to closely emulate the distribution, variety, and context of PII and sensitive data as it naturally occurs in online environments. Our data generation pipeline begins with the construction of internally consistent, multi-attribute human profiles using constrained selection to reflect real-world demographics such as education, health attributes, financial status, etc. Using a combination of zero-shot prompting and OpenAI o3-mini, we generate diverse content types - including wiki-style articles, social media posts, forum discussions, online reviews, comments, and marketplace listings - each embedding realistic, contextually appropriate PII and other sensitive information. We validate the utility of PANORAMA by fine-tuning the Mistral-7B model on 1x, 5x, 10x, and 25x data replication rates with a subset of data and measure PII memorization rates - revealing not only consistent increases with repetition but also variation across content types, highlighting PANORAMA's ability to model how memorization risks differ by context. Our dataset and code are publicly available, providing a much-needed resource for privacy risk assessment, model auditing, and the development of privacy-preserving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12237v1">From Shots to Stories: LLM-Assisted Video Editing with Unified Language Representations</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable reasoning and generalization capabilities in video understanding; however, their application in video editing remains largely underexplored. This paper presents the first systematic study of LLMs in the context of video editing. To bridge the gap between visual information and language-based reasoning, we introduce L-Storyboard, an intermediate representation that transforms discrete video shots into structured language descriptions suitable for LLM processing. We categorize video editing tasks into Convergent Tasks and Divergent Tasks, focusing on three core tasks: Shot Attributes Classification, Next Shot Selection, and Shot Sequence Ordering. To address the inherent instability of divergent task outputs, we propose the StoryFlow strategy, which converts the divergent multi-path reasoning process into a convergent selection mechanism, effectively enhancing task accuracy and logical coherence. Experimental results demonstrate that L-Storyboard facilitates a more robust mapping between visual information and language descriptions, significantly improving the interpretability and privacy protection of video editing tasks. Furthermore, StoryFlow enhances the logical consistency and output stability in Shot Sequence Ordering, underscoring the substantial potential of LLMs in intelligent video editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12249v2">LLMs are not Zero-Shot Reasoners for Biomedical Information Extraction</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly adopted for applications in healthcare, reaching the performance of domain experts on tasks such as question answering and document summarisation. Despite their success on these tasks, it is unclear how well LLMs perform on tasks that are traditionally pursued in the biomedical domain, such as structured information extraction. To bridge this gap, in this paper, we systematically benchmark LLM performance in Medical Classification and Named Entity Recognition (NER) tasks. We aim to disentangle the contribution of different factors to the performance, particularly the impact of LLMs' task knowledge and reasoning capabilities, their (parametric) domain knowledge, and addition of external knowledge. To this end, we evaluate various open LLMs - including BioMistral and Llama-2 models - on a diverse set of biomedical datasets, using standard prompting, Chain of-Thought (CoT) and Self Consistency based reasoning as well as Retrieval-Augmented Generation (RAG) with PubMed and Wikipedia corpora. Counter intuitively, our results reveal that standard prompting consistently outperforms more complex techniques across both tasks, laying bare the limitations in the current application of CoT, self-consistency and RAG in the biomedical domain. Our findings suggest that advanced prompting methods developed for knowledge- or reasoning-intensive tasks, such as CoT or RAG, are not easily portable to biomedical tasks where precise structured outputs are required. This highlights the need for more effective integration of external knowledge and reasoning mechanisms in LLMs to enhance their performance in real-world biomedical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01533v3">Enhancing LLM Evaluations: The Garbling Trick</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly powerful, traditional evaluation metrics tend to saturate, making it challenging to distinguish between models. We propose a general method to transform existing LLM evaluations into a series of progressively more difficult tasks. These enhanced evaluations emphasize reasoning capabilities and can reveal relative performance differences that are not apparent in the original assessments. To demonstrate the effectiveness of our approach, we create a new multiple-choice test corpus, extend it into a family of evaluations, and assess a collection of LLMs. Our results offer insights into the comparative abilities of these models, particularly highlighting the differences between base LLMs and more recent "reasoning" models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05831v2">Leveraging Robust Optimization for LLM Alignment under Distribution Shifts</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Preference alignment methods are increasingly critical for steering large language models (LLMs) to generate outputs consistent with human values. While recent approaches often rely on synthetic data generated by LLMs for scalability and cost-efficiency reasons, this reliance can introduce distribution shifts that undermine the nuanced representation of human preferences needed for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment despite such shifts. Our approach first leverages well-learned classifiers to assign a calibration value to each training sample, quantifying its alignment with the target human-preferred distribution. These values are then incorporated into a robust optimization objective that minimizes the worst-case loss over regions of the data space most relevant to human preferences. By explicitly focusing optimization on the target distribution, our approach mitigates the impact of distributional mismatch and improves the generation of responses that better reflect intended values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12225v1">Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      High-quality reward models are crucial for unlocking the reasoning potential of large language models (LLMs), with best-of-N voting demonstrating significant performance gains. However, current reward models, which typically operate on the textual output of LLMs, are computationally expensive and parameter-heavy, limiting their real-world applications. We introduce the Efficient Linear Hidden State Reward (ELHSR) model - a novel, highly parameter-efficient approach that leverages the rich information embedded in LLM hidden states to address these issues. ELHSR systematically outperform baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training. ELHSR also achieves orders-of-magnitude efficiency improvement with significantly less time and fewer FLOPs per sample than baseline reward models. Moreover, ELHSR exhibits robust performance even when trained only on logits, extending its applicability to some closed-source LLMs. In addition, ELHSR can also be combined with traditional reward models to achieve additional performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12212v1">Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12201v1">How Reliable is Multilingual LLM-as-a-Judge?</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has emerged as a popular evaluation strategy, where advanced large language models assess generation results in alignment with human instructions. While these models serve as a promising alternative to human annotators, their reliability in multilingual evaluation remains uncertain. To bridge this gap, we conduct a comprehensive analysis of multilingual LLM-as-a-Judge. Specifically, we evaluate five models from different model families across five diverse tasks involving 25 languages. Our findings reveal that LLMs struggle to achieve consistent judgment results across languages, with an average Fleiss' Kappa of approximately 0.3, and some models performing even worse. To investigate the cause of inconsistency, we analyze various influencing factors. We observe that consistency varies significantly across languages, with particularly poor performance in low-resource languages. Additionally, we find that neither training on multilingual data nor increasing model scale directly improves judgment consistency. These findings suggest that LLMs are not yet reliable for evaluating multilingual predictions. We finally propose an ensemble strategy which improves the consistency of the multilingual judge in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12188v1">LLM-DSE: Searching Accelerator Parameters with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-18
    </div>
    <details class="paper-abstract">
      Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample efficiency.We present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: https://github.com/Nozidoali/LLM-DSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12185v1">EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16330v2">Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) excel in many domains, their complexity and scale challenge deployment in resource-limited environments. Current compression techniques, such as parameter pruning, often fail to effectively utilize the knowledge from pruned parameters. To address these challenges, we propose Manifold-Based Knowledge Alignment and Layer Merging Compression (MKA), a novel approach that uses manifold learning and the Normalized Pairwise Information Bottleneck (NPIB) measure to merge similar layers, reducing model size while preserving essential performance. We evaluate MKA on multiple benchmark datasets and various LLMs. Our findings show that MKA not only preserves model performance but also achieves substantial compression ratios, outperforming traditional pruning methods. Moreover, when coupled with quantization, MKA delivers even greater compression. Specifically, on the MMLU dataset using the Llama3-8B model, MKA achieves a compression ratio of 43.75% with a minimal performance decrease of only 2.82\%. The proposed MKA method offers a resource-efficient and performance-preserving model compression technique for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11821v1">Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      This paper investigates approaches to enhance the reasoning capabilities of Large Language Model (LLM) agents using Reinforcement Learning (RL). Specifically, we focus on multi-turn tool-use scenarios, which can be naturally modeled as Markov Decision Processes (MDPs). While existing approaches often train multi-turn LLM agents with trajectory-level advantage estimation in bandit settings, they struggle with turn-level credit assignment across multiple decision steps, limiting their performance on multi-turn reasoning tasks. To address this, we introduce a fine-grained turn-level advantage estimation strategy to enable more precise credit assignment in multi-turn agent interactions. The strategy is general and can be incorporated into various RL algorithms such as Group Relative Preference Optimization (GRPO). Our experimental evaluation on multi-turn reasoning and search-based tool-use tasks with GRPO implementations highlights the effectiveness of the MDP framework and the turn-level credit assignment in advancing the multi-turn reasoning capabilities of LLM agents in complex decision-making settings. Our method achieves 100% success in tool execution and 50% accuracy in exact answer matching, significantly outperforming baselines, which fail to invoke tools and achieve only 20-30% exact match accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04364v2">Benchmarking LLMs' Swarm intelligence</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16918v2">OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Optimization plays a vital role in scientific research and practical applications. However, formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce OptimAI, a framework for solving Optimization problems described in natural language by leveraging LLM-powered AI agents, and achieve superior performance over current state-of-the-art methods. Our framework is built upon the following key roles: (1) a formulator that translates natural language problem descriptions into precise mathematical formulations; (2) a planner that constructs a high-level solution strategy prior to execution; and (3) a coder and a code critic capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, and our experiments confirm that combining diverse models leads to performance gains. Our approach attains 88.1% accuracy on the NLP4LP dataset and 82.3% on the Optibench dataset, reducing error rates by 58% and 52%, respectively, over prior best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20015v2">Efficient Indirect LLM Jailbreak via Multimodal-LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19128v3">Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 On going work. Codes are released at https://github.com/zyc140345/FedAMoLE
    </div>
    <details class="paper-abstract">
      Large-scale instruction data is essential for aligning pretrained Large Language Models (LLMs) with human instructions, but may contain sensitive information that hinders its public sharing. Federated Learning (FL) enables collaborative fine-tuning of LLMs without accessing raw data. However, existing approaches to federated LLM fine-tuning usually adopt a uniform model architecture, making it hard to fit highly heterogeneous client-side data in varying domains and formats. To address this, we propose FedAMoLE, a lightweight personalized FL framework that enables data-driven heterogeneous model architectures. This framework features a heterogeneous mixture of LoRA experts module for aggregating architecturally heterogeneous models and a reverse selection-based expert assignment strategy that optimizes model architectures based on data distributions. Experiments across five scenarios show that FedAMoLE improves client-side performance by an average of 5.14% compared to existing approaches while maintaining scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11795v1">The Effects of Demographic Instructions on LLM Personas</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 Accepted at SIGIR'25, Padua, Italy
    </div>
    <details class="paper-abstract">
      Social media platforms must filter sexist content in compliance with governmental regulations. Current machine learning approaches can reliably detect sexism based on standardized definitions, but often neglect the subjective nature of sexist language and fail to consider individual users' perspectives. To address this gap, we adopt a perspectivist approach, retaining diverse annotations rather than enforcing gold-standard labels or their aggregations, allowing models to account for personal or group-specific views of sexism. Using demographic data from Twitter, we employ large language models (LLMs) to personalize the identification of sexism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11772v1">LAMP: Extracting Locally Linear Decision Surfaces from LLM World Models</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      We introduce \textbf{LAMP} (\textbf{L}inear \textbf{A}ttribution \textbf{M}apping \textbf{P}robe), a method that shines light onto a black-box language model's decision surface and studies how reliably a model maps its stated reasons to its predictions through a locally linear model approximating the decision surface. LAMP treats the model's own self-reported explanations as a coordinate system and fits a locally linear surrogate that links those weights to the model's output. By doing so, it reveals which stated factors steer the model's decisions, and by how much. We apply LAMP to three tasks: \textit{sentiment analysis}, \textit{controversial-topic detection}, and \textit{safety-prompt auditing}. Across these tasks, LAMP reveals that many LLMs exhibit locally linear decision landscapes. In addition, these surfaces correlate with human judgments on explanation quality and, on a clinical case-file data set, aligns with expert assessments. Since LAMP operates without requiring access to model gradients, logits, or internal activations, it serves as a practical and lightweight framework for auditing proprietary language models, and enabling assessment of whether a model behaves consistently with the explanations it provides.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11765v1">OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited. In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06321v2">Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt. Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design. Code can be found in https://github.com/zch65458525/L2T.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10996v2">RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance on knowledge-intensive tasks, yet they often struggle with multi-step reasoning due to the unstructured nature of retrieved context. While retrieval-augmented generation (RAG) methods provide external information, the lack of explicit organization among retrieved passages limits their effectiveness, leading to brittle reasoning pathways. Recent interpretability studies highlighting the importance of structured intermediate reasoning further align with this perspective. We propose Retrieval-And-Structuring (RAS), a framework that dynamically constructs query-specific knowledge graphs through iterative retrieval and structured knowledge building. RAS interleaves targeted retrieval planning with incremental graph construction, enabling models to assemble and reason over evolving knowledge structures tailored to each query. On seven knowledge-intensive benchmarks, RAS consistently outperforms strong baselines, achieving up to 6.4% and 7.0% gains with open-source and proprietary LLMs, respectively. Our results demonstrate that dynamic, query-specific knowledge structuring offers a robust path to improving reasoning accuracy and robustness in language model generation. Our data and code can be found at https://github.com/pat-jj/RAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v2">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 9 pages, preprint
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12135v1">LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{https://github.com/choukrani/llm-babybench}{\text{GitHub}}$, $\href{https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench}{\text{HuggingFace}}$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12118v1">Do Code LLMs Do Static Analysis?</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 25 pages, 2 figures, Preprint in review
    </div>
    <details class="paper-abstract">
      This paper investigates code LLMs' capability of static analysis during code intelligence tasks such as code summarization and generation. Code LLMs are now household names for their abilities to do some programming tasks that have heretofore required people. The process that people follow to do programming tasks has long been understood to require static analysis. For example, human programmers navigate the call graph of large programs to comprehend the different parts of those programs. Education in programming includes static analysis under the assumption that better static analysis skills beget better programming. Yet while popular culture is replete with anthropomorphic references such as LLM "reasoning", in fact code LLMs could exhibit a wholly alien thought process to humans. This paper studies the specific question of static analysis by code LLMs. We use three different static analysis tasks (callgraph generation, AST generation, and dataflow generation) and three different code intelligence tasks (code generation, summarization, and translation) with two different open-source models (Gemini and GPT-4o) and closed-source models (CodeLlaMA and Jam) as our experiments. We found that LLMs show poor performance on static analysis tasks and that pretraining on the static analysis tasks does not generalize to better performance on the code intelligence tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12104v1">The Impact of Emerging Phishing Threats: Assessing Quishing and LLM-generated Phishing Emails against Organizations</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 Accepted to AsiaCCS'25
    </div>
    <details class="paper-abstract">
      Modern organizations are persistently targeted by phishing emails. Despite advances in detection systems and widespread employee training, attackers continue to innovate, posing ongoing threats. Two emerging vectors stand out in the current landscape: QR-code baits and LLM-enabled pretexting. Yet, little is known about the effectiveness of current defenses against these attacks, particularly when it comes to real-world impact on employees. This gap leaves uncertainty around to what extent related countermeasures are justified or needed. Our work addresses this issue. We conduct three phishing simulations across organizations of varying sizes -- from small-medium businesses to a multinational enterprise. In total, we send over 71k emails targeting employees, including: a "traditional" phishing email with a click-through button; a nearly-identical "quishing" email with a QR code instead; and a phishing email written with the assistance of an LLM and open-source intelligence. Our results show that quishing emails have the same effectiveness as traditional phishing emails at luring users to the landing webpage -- which is worrying, given that quishing emails are much harder to identify even by operational detectors. We also find that LLMs can be very good "social engineers": in one company, over 30% of the emails opened led to visiting the landing webpage -- a rate exceeding some prior benchmarks. Finally, we complement our study by conducting a survey across the organizations' employees, measuring their "perceived" phishing awareness. Our findings suggest a correlation between higher self-reported awareness and organizational resilience to phishing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12100v1">Improving Fairness in LLMs Through Testing-Time Adversaries</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) push the bound-aries in natural language processing and generative AI, driving progress across various aspects of modern society. Unfortunately, the pervasive issue of bias in LLMs responses (i.e., predictions) poses a significant and open challenge, hindering their application in tasks involving ethical sensitivity and responsible decision-making. In this work, we propose a straightforward, user-friendly and practical method to mitigate such biases, enhancing the reliability and trustworthiness of LLMs. Our method creates multiple variations of a given sentence by modifying specific attributes and evaluates the corresponding prediction behavior compared to the original, unaltered, prediction/sentence. The idea behind this process is that critical ethical predictions often exhibit notable inconsistencies, indicating the presence of bias. Unlike previous approaches, our method relies solely on forward passes (i.e., testing-time adversaries), eliminating the need for training, fine-tuning, or prior knowledge of the training data distribution. Through extensive experiments on the popular Llama family, we demonstrate the effectiveness of our method in improving various fairness metrics, focusing on the reduction of disparities in how the model treats individuals from different racial groups. Specifically, using standard metrics, we improve the fairness in Llama3 in up to 27 percentage points. Overall, our approach significantly enhances fairness, equity, and reliability in LLM-generated results without parameter tuning or training data modifications, confirming its effectiveness in practical scenarios. We believe our work establishes an important step toward enabling the use of LLMs in tasks that require ethical considerations and responsible decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05605v3">Evolving LLMs' Self-Refinement Capability via Iterative Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated remarkable general performance, enabling smaller models to achieve capabilities comparable to their larger counterparts remains a critical challenge. For humans, iterative refinement of problem analysis and responses is a common strategy to enhance answer quality. However, we observe that existing LLMs exhibit limited ability to refine their outputs for quality improvement. In this paper, we first investigate mechanisms to unlock and progressively enhance self-refinement ability in smaller models within an iterative preference optimization framework, aiming to bridge the performance gap with larger models. To this end, we propose EVOLVE, a novel post-training and inference framework that iteratively integrates preference training with self-refinement-driven data collection. During training, EVOLVE strengthens the model's direct question-answering ability while simultaneously unlocking its self-refinement potential. At inference, the framework leverages this capability to generate progressively refined responses, which are filtered to construct datasets for subsequent rounds of preference training. Experiments demonstrate EVOLVE's exceptional performance: when applied to Llama-3.1-8B base model and under the self-refinement setting, it surpasses state-of-the-art models including Llama-3.1-405B-Instruct and GPT-4o, achieving a 62.3% length-controlled win rate and 63.3% raw win rate on AlpacaEval 2, along with a 50.3% win rate on Arena-Hard. Furthermore, EVOLVE consistently enhances performance on mathematical reasoning tasks like GSM8K and MATH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03261v2">Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Multiple previous studies have reported suboptimal performance of LLMs in biomedical text mining. By analyzing failure patterns in these evaluations, we identified three primary challenges for LLMs in biomedical corpora: (1) LLMs fail to learn implicit dataset-specific nuances from supervised data, (2) The common formatting requirements of discriminative tasks limit the reasoning capabilities of LLMs particularly for LLMs that lack test-time compute, and (3) LLMs struggle to adhere to annotation guidelines and match exact schemas, which hinders their ability to understand detailed annotation requirements which is essential in biomedical annotation workflow. We experimented with prompt engineering techniques targeted to the above issues, and developed a pipeline that dynamically extracts instructions from annotation guidelines. Our results show that frontier LLMs can approach or surpass the performance of SOTA BERT-based models with minimal reliance on manually annotated data and without fine-tuning. Furthermore, we performed model distillation on a closed-source LLM, demonstrating that a BERT model trained exclusively on synthetic data annotated by LLMs can also achieve a practical performance. Based on these findings, we explored the feasibility of partially replacing manual annotation with LLMs in production scenarios for biomedical text mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22388v2">Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the future. DSDBench is publicly available at github.com/KevinCL16/DSDBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12060v1">Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 Acccepted by ACL 2025 Findings, 21 pages, 9 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at https://github.com/NJUNLP/SAGE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12058v1">Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 28 pages, 7 figures, 3 tables. Includes expanded appendix & full score matrices. Dataset & code: HF Hub + GitHub + Pypi links in abstract. Core data and code Apache-2.0; synthetic packs eval-only
    </div>
    <details class="paper-abstract">
      Tiny QA Benchmark++ (TQB++) presents an ultra-lightweight, multilingual smoke-test suite designed to give large-language-model (LLM) pipelines a unit-test style safety net dataset that runs in seconds with minimal cost. Born out of the tight feedback-loop demands building the Comet Opik prompt-optimization SDK, where waiting on heavyweight benchmarks breaks developer flow. TQB++ couples a 52-item English gold set (less than 20 kB) with a tiny synthetic-data generator pypi package built on provider-agnostic LiteLLM. The generator lets practitioners mint their own tiny packs in any language, domain, or difficulty, while ten ready-made packs already cover Arabic, Chinese, French, German, Japanese, Korean, Portuguese, Russian, Spanish, and Turkish. Every dataset ships with Croissant metadata and plug-and-play files for OpenAI-Evals, LangChain, and standard CI tools, so teams can drop deterministic micro-benchmarks directly into pull-request gates, prompt-engineering loops, and production dashboards without touching GPU budgets. A complete TQB++ run adds only a few seconds to pipeline latency yet reliably flags prompt-template errors, tokenizer drift, and fine-tuning side-effects long before full-scale suites like MMLU or BIG-Bench would finish configuring. The entire framework is released to accelerate continuous, resource-efficient quality assurance across the generative-AI ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v5">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12054v1">GenderBench: Evaluation Suite for Gender Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      We present GenderBench -- a comprehensive evaluation suite designed to measure gender biases in LLMs. GenderBench includes 14 probes that quantify 19 gender-related harmful behaviors exhibited by LLMs. We release GenderBench as an open-source and extensible library to improve the reproducibility and robustness of benchmarking across the field. We also publish our evaluation of 12 LLMs. Our measurements reveal consistent patterns in their behavior. We show that LLMs struggle with stereotypical reasoning, equitable gender representation in generated texts, and occasionally also with discriminatory behavior in high-stakes scenarios, such as hiring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12043v1">MoL for LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving General Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Although LLMs perform well in general tasks, domain-specific applications suffer from hallucinations and accuracy limitations. CPT approaches encounter two key issues: (1) domain-biased data degrades general language skills, and (2) improper corpus-mixture ratios limit effective adaptation. To address these, we propose a novel framework, Mixture of Losses (MoL), which decouples optimization objectives for domain-specific and general corpora. Specifically, cross-entropy (CE) loss is applied to domain data to ensure knowledge acquisition, while Kullback-Leibler (KL) divergence aligns general-corpus training with the base model's foundational capabilities. This dual-loss architecture preserves universal skills while enhancing domain expertise, avoiding catastrophic forgetting. Empirically, we validate that a 1:1 domain-to-general corpus ratio optimally balances training and overfitting without the need for extensive tuning or resource-intensive experiments. Furthermore, our experiments demonstrate significant performance gains compared to traditional CPT approaches, which often suffer from degradation in general language capabilities; our model achieves 27.9% higher accuracy on the Math-500 benchmark in the non-think reasoning mode, and an impressive 83.3% improvement on the challenging AIME25 subset in the think mode, underscoring the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12038v1">Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 ICML 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential as general-purpose AI assistants across various domains. To fully leverage this potential in specific applications, many companies provide fine-tuning API services, enabling users to upload their own data for LLM customization. However, fine-tuning services introduce a new safety threat: user-uploaded data, whether harmful or benign, can break the model's alignment, leading to unsafe outputs. Moreover, existing defense methods struggle to address the diversity of fine-tuning datasets (e.g., varying sizes, tasks), often sacrificing utility for safety or vice versa. To address this issue, we propose Safe Delta, a safety-aware post-training defense method that adjusts the delta parameters (i.e., the parameter change before and after fine-tuning). Specifically, Safe Delta estimates the safety degradation, selects delta parameters to maximize utility while limiting overall safety loss, and applies a safety compensation vector to mitigate residual safety loss. Through extensive experiments on four diverse datasets with varying settings, our approach consistently preserves safety while ensuring that the utility gain from benign datasets remains unaffected.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12031v1">LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have sparked considerable interest in automated theorem proving and a prominent line of research integrates stepwise LLM-based provers into tree search. In this paper, we introduce a novel proof-state exploration approach for training data synthesis, designed to produce diverse tactics across a wide range of intermediate proof states, thereby facilitating effective one-shot fine-tuning of LLM as the policy model. We also propose an adaptive beam size strategy, which effectively takes advantage of our data synthesis method and achieves a trade-off between exploration and exploitation during tree search. Evaluations on the MiniF2F and ProofNet benchmarks demonstrate that our method outperforms strong baselines under the stringent Pass@1 metric, attaining an average pass rate of $60.74\%$ on MiniF2F and $21.18\%$ on ProofNet. These results underscore the impact of large-scale synthetic data in advancing automated theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19162v2">SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 Project webpage: https://chen-judge.github.io/SPC/
    </div>
    <details class="paper-abstract">
      Evaluating the step-by-step reliability of large language model (LLM) reasoning, such as Chain-of-Thought, remains challenging due to the difficulty and cost of obtaining high-quality step-level supervision. In this paper, we introduce Self-Play Critic (SPC), a novel approach where a critic model evolves its ability to assess reasoning steps through adversarial self-play games, eliminating the need for manual step-level annotation. SPC involves fine-tuning two copies of a base model to play two roles, namely a "sneaky generator" that deliberately produces erroneous steps designed to be difficult to detect, and a "critic" that analyzes the correctness of reasoning steps. These two models engage in an adversarial game in which the generator aims to fool the critic, while the critic model seeks to identify the generator's errors. Using reinforcement learning based on the game outcomes, the models iteratively improve; the winner of each confrontation receives a positive reward and the loser receives a negative reward, driving continuous self-evolution. Experiments on three reasoning process benchmarks (ProcessBench, PRM800K, DeltaBench) demonstrate that our SPC progressively enhances its error detection capabilities (e.g., accuracy increases from 70.8% to 77.7% on ProcessBench) and surpasses strong baselines, including distilled R1 model. Furthermore, SPC can guide the test-time search of diverse LLMs and significantly improve their mathematical reasoning performance on MATH500 and AIME2024, surpassing those guided by state-of-the-art process reward models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12001v1">Interactional Fairness in LLM Multi-Agent Systems: An Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in multi-agent systems, questions of fairness should extend beyond resource distribution and procedural design to include the fairness of how agents communicate. Drawing from organizational psychology, we introduce a novel framework for evaluating Interactional fairness encompassing Interpersonal fairness (IF) and Informational fairness (InfF) in LLM-based multi-agent systems (LLM-MAS). We extend the theoretical grounding of Interactional Fairness to non-sentient agents, reframing fairness as a socially interpretable signal rather than a subjective experience. We then adapt established tools from organizational justice research, including Colquitt's Organizational Justice Scale and the Critical Incident Technique, to measure fairness as a behavioral property of agent interaction. We validate our framework through a pilot study using controlled simulations of a resource negotiation task. We systematically manipulate tone, explanation quality, outcome inequality, and task framing (collaborative vs. competitive) to assess how IF influences agent behavior. Results show that tone and justification quality significantly affect acceptance decisions even when objective outcomes are held constant. In addition, the influence of IF vs. InfF varies with context. This work lays the foundation for fairness auditing and norm-sensitive alignment in LLM-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11995v1">Unveiling Knowledge Utilization Mechanisms in LLM-based Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 SIGIR 2025
    </div>
    <details class="paper-abstract">
      Considering the inherent limitations of parametric knowledge in large language models (LLMs), retrieval-augmented generation (RAG) is widely employed to expand their knowledge scope. Since RAG has shown promise in knowledge-intensive tasks like open-domain question answering, its broader application to complex tasks and intelligent assistants has further advanced its utility. Despite this progress, the underlying knowledge utilization mechanisms of LLM-based RAG remain underexplored. In this paper, we present a systematic investigation of the intrinsic mechanisms by which LLMs integrate internal (parametric) and external (retrieved) knowledge in RAG scenarios. Specially, we employ knowledge stream analysis at the macroscopic level, and investigate the function of individual modules at the microscopic level. Drawing on knowledge streaming analyses, we decompose the knowledge utilization process into four distinct stages within LLM layers: knowledge refinement, knowledge elicitation, knowledge expression, and knowledge contestation. We further demonstrate that the relevance of passages guides the streaming of knowledge through these stages. At the module level, we introduce a new method, knowledge activation probability entropy (KAPE) for neuron identification associated with either internal or external knowledge. By selectively deactivating these neurons, we achieve targeted shifts in the LLM's reliance on one knowledge source over the other. Moreover, we discern complementary roles for multi-head attention and multi-layer perceptron layers during knowledge formation. These insights offer a foundation for improving interpretability and reliability in retrieval-augmented LLMs, paving the way for more robust and transparent generative solutions in knowledge-intensive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11978v1">LLM-guided DRL for Multi-tier LEO Satellite Networks with Hybrid FSO/RF Links</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 This paper has been submitted to IEEE JSAC
    </div>
    <details class="paper-abstract">
      Despite significant advancements in terrestrial networks, inherent limitations persist in providing reliable coverage to remote areas and maintaining resilience during natural disasters. Multi-tier networks with low Earth orbit (LEO) satellites and high-altitude platforms (HAPs) offer promising solutions, but face challenges from high mobility and dynamic channel conditions that cause unstable connections and frequent handovers. In this paper, we design a three-tier network architecture that integrates LEO satellites, HAPs, and ground terminals with hybrid free-space optical (FSO) and radio frequency (RF) links to maximize coverage while maintaining connectivity reliability. This hybrid approach leverages the high bandwidth of FSO for satellite-to-HAP links and the weather resilience of RF for HAP-to-ground links. We formulate a joint optimization problem to simultaneously balance downlink transmission rate and handover frequency by optimizing network configuration and satellite handover decisions. The problem is highly dynamic and non-convex with time-coupled constraints. To address these challenges, we propose a novel large language model (LLM)-guided truncated quantile critics algorithm with dynamic action masking (LTQC-DAM) that utilizes dynamic action masking to eliminate unnecessary exploration and employs LLMs to adaptively tune hyperparameters. Simulation results demonstrate that the proposed LTQC-DAM algorithm outperforms baseline algorithms in terms of convergence, downlink transmission rate, and handover frequency. We also reveal that compared to other state-of-the-art LLMs, DeepSeek delivers the best performance through gradual, contextually-aware parameter adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08492v2">Achieving Scalable Robot Autonomy via neurosymbolic planning using lightweight local LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 18 pages, 3 figures, 4 tables, accepted at IAS 2025
    </div>
    <details class="paper-abstract">
      PDDL-based symbolic task planning remains pivotal for robot autonomy yet struggles with dynamic human-robot collaboration due to scalability, re-planning demands, and delayed plan availability. Although a few neurosymbolic frameworks have previously leveraged LLMs such as GPT-3 to address these challenges, reliance on closed-source, remote models with limited context introduced critical constraints: third-party dependency, inconsistent response times, restricted plan length and complexity, and multi-domain scalability issues. We present Gideon, a novel framework that enables the transition to modern, smaller, local LLMs with extended context length. Gideon integrates a novel problem generator to systematically generate large-scale datasets of realistic domain-problem-plan tuples for any domain, and adapts neurosymbolic planning for local LLMs, enabling on-device execution and extended context for multi-domain support. Preliminary experiments in single-domain scenarios performed on Qwen-2.5 1.5B and trained on 8k-32k samples, demonstrate a valid plan percentage of 66.1% (32k model) and show that the figure can be further scaled through additional data. Multi-domain tests on 16k samples yield an even higher 70.6% planning validity rate, proving extensibility across domains and signaling that data variety can have a positive effect on learning efficiency. Although long-horizon planning and reduced model size make Gideon training much less efficient than baseline models based on larger LLMs, the results are still significant considering that the trained model is about 120x smaller than baseline and that significant advantages can be achieved in inference efficiency, scalability, and multi-domain adaptability, all critical factors in human-robot collaboration. Training inefficiency can be mitigated by Gideon's streamlined data generation pipeline.
    </details>
</div>
