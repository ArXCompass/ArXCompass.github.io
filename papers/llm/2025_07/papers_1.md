# llm - 2025_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23773v1">SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning Architecture with LLM-Based World Model</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      AI agents built on large language models (LLMs) hold enormous promise, but current practice focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also suffers from the fundamental limitations of autoregressive LLMs. On the other hand, humans are general agents who reason by mentally simulating the outcomes of their actions and plans. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of optimal agent in any environment, \modelname overcomes the limitations of autoregressive reasoning by introducing a world model for planning via simulation. The generalized world model is implemented using LLM, which can flexibly plan in a wide range of environments using the concept-rich latent space of natural language. Experiments on difficult web browsing tasks show that \modelname improves the success of flight search from 0\% to 32.2\%. World-model-based planning, in particular, shows consistent advantage of up to 124\% over autoregressive planning, demonstrating the advantage of world model simulation as a reasoning paradigm. We are excited about the possibility for training a single, general agent model based on LLMs that can act superintelligently in all environments. To start, we make SimuRA, a web-browsing agent built on \modelname with pretrained LLMs, available as a research demo for public testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08184v3">Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Accepted at ICML 2025 Workshop on Long Context Foundation Models (ICFM). Code: https://github.com/zhuangziGiantfish/Unable-to-Forget
    </div>
    <details class="paper-abstract">
      Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23701v1">TextQuests: How Good are LLMs at Text-Based Video Games?</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Evaluating AI agents within complex, interactive environments that mirror real-world challenges is critical for understanding their practical capabilities. While existing agent benchmarks effectively assess skills like tool use or performance on structured tasks, they often do not fully capture an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context. To spur the development of agents capable of more robust intrinsic reasoning over long horizons, we introduce TextQuests, a benchmark based on the Infocom suite of interactive fiction games. These text-based adventures, which can take human players over 30 hours and require hundreds of precise actions to solve, serve as an effective proxy for evaluating AI agents on focused, stateful tasks. The benchmark is specifically designed to assess an LLM agent's capacity for self-contained problem-solving by precluding the use of external tools, thereby focusing on intrinsic long-context reasoning capabilities in an exploratory environment characterized by the need for trial-and-error learning and sustained problem-solving within a single interactive session. We release TextQuests at https://textquests.ai.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23694v1">A survey of multi-agent geosimulation methodologies: from ABM to LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 20 pages, 1 table
    </div>
    <details class="paper-abstract">
      We provide a comprehensive examination of agent-based approaches that codify the principles and linkages underlying multi-agent systems, simulations, and information systems. Based on two decades of study, this paper confirms a framework intended as a formal specification for geosimulation platforms. Our findings show that large language models (LLMs) can be effectively incorporated as agent components if they follow a structured architecture specific to fundamental agent activities such as perception, memory, planning, and action. This integration is precisely consistent with the architecture that we formalize, providing a solid platform for next-generation geosimulation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v2">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23633v1">MemoCue: Empowering LLM-Based Agents for Human Memory Recall via Strategy-Guided Querying</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Agent-assisted memory recall is one critical research problem in the field of human-computer interaction. In conventional methods, the agent can retrieve information from its equipped memory module to help the person recall incomplete or vague memories. The limited size of memory module hinders the acquisition of complete memories and impacts the memory recall performance in practice. Memory theories suggest that the person's relevant memory can be proactively activated through some effective cues. Inspired by this, we propose a novel strategy-guided agent-assisted memory recall method, allowing the agent to transform an original query into a cue-rich one via the judiciously designed strategy to help the person recall memories. To this end, there are two key challenges. (1) How to choose the appropriate recall strategy for diverse forgetting scenarios with distinct memory-recall characteristics? (2) How to obtain the high-quality responses leveraging recall strategies, given only abstract and sparsely annotated strategy patterns? To address the challenges, we propose a Recall Router framework. Specifically, we design a 5W Recall Map to classify memory queries into five typical scenarios and define fifteen recall strategy patterns across the corresponding scenarios. We then propose a hierarchical recall tree combined with the Monte Carlo Tree Search algorithm to optimize the selection of strategy and the generation of strategy responses. We construct an instruction tuning dataset and fine-tune multiple open-source large language models (LLMs) to develop MemoCue, an agent that excels in providing memory-inspired responses. Experiments on three representative datasets show that MemoCue surpasses LLM-based methods by 17.74% in recall inspiration. Further human evaluation highlights its advantages in memory-recall applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23611v1">LLM-Based Identification of Infostealer Infection Vectors from Screenshots: The Case of Aurora</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Infostealers exfiltrate credentials, session cookies, and sensitive data from infected systems. With over 29 million stealer logs reported in 2024, manual analysis and mitigation at scale are virtually unfeasible/unpractical. While most research focuses on proactive malware detection, a significant gap remains in leveraging reactive analysis of stealer logs and their associated artifacts. Specifically, infection artifacts such as screenshots, image captured at the point of compromise, are largely overlooked by the current literature. This paper introduces a novel approach leveraging Large Language Models (LLMs), more specifically gpt-4o-mini, to analyze infection screenshots to extract potential Indicators of Compromise (IoCs), map infection vectors, and track campaigns. Focusing on the Aurora infostealer, we demonstrate how LLMs can process screenshots to identify infection vectors, such as malicious URLs, installer files, and exploited software themes. Our method extracted 337 actionable URLs and 246 relevant files from 1000 screenshots, revealing key malware distribution methods and social engineering tactics. By correlating extracted filenames, URLs, and infection themes, we identified three distinct malware campaigns, demonstrating the potential of LLM-driven analysis for uncovering infection workflows and enhancing threat intelligence. By shifting malware analysis from traditional log-based detection methods to a reactive, artifact-driven approach that leverages infection screenshots, this research presents a scalable method for identifying infection vectors and enabling early intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v3">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23554v1">DICE: Dynamic In-Context Example Selection in LLM Agents via Efficient Knowledge Transfer</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Large language model-based agents, empowered by in-context learning (ICL), have demonstrated strong capabilities in complex reasoning and tool-use tasks. However, existing works have shown that the effectiveness of ICL is highly sensitive to the choice of demonstrations, with suboptimal examples often leading to unstable or degraded performance. While prior work has explored example selection, including in some agentic or multi-step settings, existing approaches typically rely on heuristics or task-specific designs and lack a general, theoretically grounded criterion for what constitutes an effective demonstration across reasoning steps. Therefore, it is non-trivial to develop a principled, general-purpose method for selecting demonstrations that consistently benefit agent performance. In this paper, we address this challenge with DICE, Dynamic In-Context Example Selection for LLM Agents, a theoretically grounded ICL framework for agentic tasks that selects the most relevant demonstrations at each step of reasoning. Our approach decomposes demonstration knowledge into transferable and non-transferable components through a causal lens, showing how the latter can introduce spurious dependencies that impair generalization. We further propose a stepwise selection criterion with a formal guarantee of improved agent performance. Importantly, DICE is a general, framework-agnostic solution that can be integrated as a plug-in module into existing agentic frameworks without any additional training cost. Extensive experiments across diverse domains demonstrate our method's effectiveness and generality, highlighting the importance of principled, context-aware demo selection for robust and efficient LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v4">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23536v1">From LLMs to Edge: Parameter-Efficient Fine-Tuning on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Parameter-efficient fine-tuning (PEFT) methods reduce the computational costs of updating deep learning models by minimizing the number of additional parameters used to adapt a model to a down- stream task. While extensively researched in large language models (LLMs), their application to smaller models used on edge devices, such as convolutional neural networks, remains underexplored. This paper benchmarks and analyzes popular PEFT methods on convolutional architectures typically deployed in resource-constrained edge environments. We evaluate LoRA, DoRA, and GaLore for updating standard and depthwise convolutional architectures to handle distribution shifts and accommodate unseen classes. We utilize recently proposed PyTorch profilers to compare the updated model performance and computational costs of these PEFT methods with traditional fine-tuning approaches. With resource efficiency in mind, we investigate their update behavior across different rank dimensions. We find that the evaluated PEFT methods are only half as memory-efficient when applied to depthwise-separable convolution architectures, compared to their efficiency with LLMs. Conversely, when targeting convolu- tional architectures optimized for edge deployment, adapter-based PEFT methods can reduce floating point operations (FLOPs) during model updates by up to 95%. These insights offer valuable guidance for selecting PEFT methods based on hardware constraints, performance requirements, and application needs. Our code is online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15621v2">LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 ICCV 2025 Workshop on What is Next in Multimodal Foundation Models
    </div>
    <details class="paper-abstract">
      Recent progress in Multimodal Large Language Models (MLLMs) has highlighted the critical roles of both the visual backbone and the underlying language model. While prior work has primarily focused on scaling these components to billions of parameters, the trade-offs between model size, architecture, and performance remain underexplored. Additionally, inconsistencies in training data and evaluation protocols have hindered direct comparisons, making it difficult to derive optimal design choices. In this paper, we introduce LLaVA-MORE, a new family of MLLMs that integrates recent language models with diverse visual backbones. To ensure fair comparisons, we employ a unified training protocol applied consistently across all architectures. Our analysis systematically explores both small- and medium-scale LLMs -- including Phi-4, LLaMA-3.1, and Gemma-2 -- to evaluate multimodal reasoning, generation, and instruction following, while examining the relationship between model size and performance. Beyond evaluating the LLM impact on final results, we conduct a comprehensive study of various visual encoders, ranging from CLIP-based architectures to alternatives such as DINOv2, SigLIP, and SigLIP2. Additional experiments investigate the effects of increased image resolution and variations in pre-training datasets. Overall, our results provide insights into the design of more effective MLLMs, offering a reproducible evaluation framework that facilitates direct comparisons and can guide future model development. Our source code and trained models are publicly available at: https://github.com/aimagelab/LLaVA-MORE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23486v1">A Novel Evaluation Benchmark for Medical LLMs: Illuminating Safety and Effectiveness in Clinical Domains</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) hold promise in clinical decision support but face major challenges in safety evaluation and effectiveness validation. We developed the Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB), a multidimensional framework built on clinical expert consensus, encompassing 30 criteria covering critical areas like critical illness recognition, guideline adherence, and medication safety, with weighted consequence measures. Thirty-two specialist physicians developed and reviewed 2,069 open-ended Q&A items aligned with these criteria, spanning 26 clinical departments to simulate real-world scenarios. Benchmark testing of six LLMs revealed moderate overall performance (average total score 57.2%, safety 54.7%, effectiveness 62.3%), with a significant 13.3% performance drop in high-risk scenarios (p < 0.0001). Domain-specific medical LLMs showed consistent performance advantages over general-purpose models, with relatively higher top scores in safety (0.912) and effectiveness (0.861). The findings of this study not only provide a standardized metric for evaluating the clinical application of medical LLMs, facilitating comparative analyses, risk exposure identification, and improvement directions across different scenarios, but also hold the potential to promote safer and more effective deployment of large language models in healthcare environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23453v1">Counterfactual Evaluation for Blind Attack Detection in LLM-based Evaluation Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      This paper investigates defenses for LLM-based evaluation systems against prompt injection. We formalize a class of threats called blind attacks, where a candidate answer is crafted independently of the true answer to deceive the evaluator. To counter such attacks, we propose a framework that augments Standard Evaluation (SE) with Counterfactual Evaluation (CFE), which re-evaluates the submission against a deliberately false ground-truth answer. An attack is detected if the system validates an answer under both standard and counterfactual conditions. Experiments show that while standard evaluation is highly vulnerable, our SE+CFE framework significantly improves security by boosting attack detection with minimal performance trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14928v3">EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Paper URL: https://aclanthology.org/2025.acl-long.1576 ;Presentation Video: https://www.youtube.com/watch?v=j63ooKE50I0
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10375v2">Mokav: Execution-driven Differential Testing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      It is essential to detect functional differences between programs in various software engineering tasks, such as automated program repair, mutation testing, and code refactoring. The problem of detecting functional differences between two programs can be reduced to searching for a difference exposing test (DET): a test input that results in different outputs on the subject programs. In this paper, we propose Mokav, a novel execution-driven tool that leverages LLMs to generate DETs. Mokav takes two versions of a program (P and Q) and an example test input. When successful, Mokav generates a valid DET, a test input that leads to provably different outputs on P and Q. Mokav iteratively prompts an LLM with a specialized prompt to generate new test inputs. At each iteration, Mokav provides execution-based feedback from previously generated tests until the LLM produces a DET. We evaluate Mokav on 1535 pairs of Python programs collected from the Codeforces competition platform and 32 pairs of programs from the QuixBugs dataset. Our experiments show that Mokav outperforms the state-of-the-art, Pynguin and Differential Prompting, by a large margin. Mokav can generate DETs for 81.7% (1,255/1535) of the program pairs in our benchmark (versus 4.9% for Pynguin and 37.3% for Differential Prompting). We demonstrate that the iterative and execution-driven feedback components of the system contribute to its high effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23410v1">Towards LLM-Enhanced Product Line Scoping</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      The idea of product line scoping is to identify the set of features and configurations that a product line should include, i.e., offer for configuration purposes. In this context, a major scoping task is to find a balance between commercial relevance and technical feasibility. Traditional product line scoping approaches rely on formal feature models and require a manual analysis which can be quite time-consuming. In this paper, we sketch how Large Language Models (LLMs) can be applied to support product line scoping tasks with a natural language interaction based scoping process. Using a working example from the smarthome domain, we sketch how LLMs can be applied to evaluate different feature model alternatives. We discuss open research challenges regarding the integration of LLMs with product line scoping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23399v1">Beyond the Cloud: Assessing the Benefits and Drawbacks of Local LLM Deployment for Translators</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models presents both opportunities and challenges for the translation field. While commercial, cloud-based AI chatbots have garnered significant attention in translation studies, concerns regarding data privacy, security, and equitable access necessitate exploration of alternative deployment models. This paper investigates the feasibility and performance of locally deployable, free language models as a viable alternative to proprietary, cloud-based AI solutions. This study evaluates three open-source models installed on CPU-based platforms and compared against commercially available online chat-bots. The evaluation focuses on functional performance rather than a comparative analysis of human-machine translation quality, an area already subject to extensive research. The platforms assessed were chosen for their accessibility and ease of use across various operating systems. While local deployment introduces its own challenges, the benefits of enhanced data control, improved privacy, and reduced dependency on cloud services are compelling. The findings of this study contribute to a growing body of knowledge concerning the democratization of AI technology and inform future research and development efforts aimed at making LLMs more accessible and practical for a wider range of users, specifically focusing on the needs of individual translators and small businesses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23386v1">Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Decoder-only large language models (LLMs) are increasingly used to build embedding models that effectively encode the semantic information of natural language texts into dense vector representations for various embedding tasks. However, many existing methods primarily focus on removing the causal attention mask in LLMs to enable bidirectional attention, potentially undermining the model's ability to extract semantic information acquired during pretraining. Additionally, leading unidirectional approaches often rely on extra input text to overcome the inherent limitations of causal attention, inevitably increasing computational costs. In this work, we propose Causal2Vec, a general-purpose embedding model tailored to enhance the performance of decoder-only LLMs without altering their original architectures or introducing significant computational overhead. Specifically, we first employ a lightweight BERT-style model to pre-encode the input text into a single Contextual token, which is then prepended to the LLM's input sequence, allowing each token to capture contextualized information even without attending to future tokens. Furthermore, to mitigate the recency bias introduced by last-token pooling and help LLMs better leverage the semantic information encoded in the Contextual token, we concatenate the last hidden states of Contextual and EOS tokens as the final text embedding. In practice, Causal2Vec achieves state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB) among models trained solely on publicly available retrieval datasets, while reducing the required sequence length by up to 85% and inference time by up to 82% compared to best-performing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23377v1">LLM4Rail: An LLM-Augmented Railway Service Consulting Platform</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly reshaped different walks of business. To meet the increasing demands for individualized railway service, we develop LLM4Rail - a novel LLM-augmented railway service consulting platform. Empowered by LLM, LLM4Rail can provide custom modules for ticketing, railway food & drink recommendations, weather information, and chitchat. In LLM4Rail, we propose the iterative "Question-Thought-Action-Observation (QTAO)" prompting framework. It meticulously integrates verbal reasoning with task-oriented actions, that is, reasoning to guide action selection, to effectively retrieve external observations relevant to railway operation and service to generate accurate responses. To provide personalized onboard dining services, we first construct the Chinese Railway Food and Drink (CRFD-25) - a publicly accessible takeout dataset tailored for railway services. CRFD-25 covers a wide range of signature dishes categorized by cities, cuisines, age groups, and spiciness levels. We further introduce an LLM-based zero-shot conversational recommender for railway catering. To address the unconstrained nature of open recommendations, the feature similarity-based post-processing step is introduced to ensure all the recommended items are aligned with CRFD-25 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23370v1">Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Pengfei Gao and Zhao Tian contributed equally to this technical report
    </div>
    <details class="paper-abstract">
      Software issue resolution is a critical challenge in software engineering and has garnered increasing attention in recent years. With the rapid advancement of large language models (LLMs), substantial progress has been made in addressing real-world software engineering tasks. Recent studies have introduced ensemble reasoning techniques to enhance the performance of LLM-based issue resolution. However, existing prompting-based methods still face limitations in effectively exploring large ensemble spaces and lack the capacity for repository-level understanding, both of which constrain their overall effectiveness. In this paper, we propose Trae Agent, the first agent-based ensemble reasoning approach for repository-level issue resolution. Trae Agent formulates our goal as an optimal solution search problem and addresses two key challenges, i.e., large ensemble spaces and repository-level understanding, through modular agents for generation, pruning, and selection. We conduct extensive experiments using three leading LLMs on the widely-adopted SWE-bench benchmark, comparing Trae Agent against four state-of-the-art ensemble reasoning techniques. Experimental results demonstrate that Trae Agent consistently achieves superior performance, with an average improvement of 10.22% over all baselines in terms of Pass@1. Trae Agent has achieved first place on the SWE-bench Verified leaderboard, with a notable Pass@1 score of 75.20%. We are pleased to release Trae Agent as an open-source project to support the research community, with all resources available at https://github.com/bytedance/trae-agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21875v2">WildSpeech-Bench: Benchmarking Audio LLMs in Natural Speech Conversation</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Recent multi-modal Large Language Models (LLMs) such as GPT-4o have demonstrated strong capabilities of direct speech interaction. However, the lack of specialized and comprehensive benchmarks for end-to-end speech LLM evaluation hinders optimizing the user experience of Audio LLMs in real-world applications. Existing evaluation methods often adapt text-based benchmarks, overlooking speech's unique characteristics and challenges, including prosody, homophones, stuttering, and differing user expectations. Here, we present a novel approach to thoroughly evaluate LLMs in practical speech conversations. We systematically curate real-world chat data relevant to spoken scenarios, introduce diversity in speaker attributes and acoustic conditions, and augment the dataset with speech-specific phenomena. We further design a query-aware evaluation method to use customized evaluation checklists and prompts to enhance the accuracy of automatic evaluation. We conduct comprehensive testing and detailed analysis of various mainstream speech models, revealing significant differences in model performance across different speech scenarios. The use of query-aware evaluation further enables a finer-grained assessment under various speech-specific scenarios. Our benchmark can provide valuable insights for speech model development and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07695v2">KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 21 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08403v6">LLMs and the Human Condition</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Edits and rearranged to provide significantly tighter writing - no change to the content. Now targeting a one day workshop on multi agent systems and LLMs
    </div>
    <details class="paper-abstract">
      Theory based AI research has had a hard time recently and the aim here is to propose a model of what LLMs are actually doing when they impress us with their language skills. The model integrates three established theories of human decision-making from philosophy, sociology, and computer science. The paper starts with the collective understanding of reasoning from the early days of AI research - primarily because that model is how we humans think we think, and is the most accessible. It then describes what is commonly thought of as "reactive systems" which is the position taken by many philosophers and indeed many contemporary AI researchers. The third component to the proposed model is from sociology and based on the idea that human intelligence is a collective skill for which individuals are merely actors. The resulting model provides an alternate view of ``mind reading'' in human communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23319v1">What's Taboo for You? - An Empirical Evaluation of LLMs Behavior Toward Sensitive Content</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Proprietary Large Language Models (LLMs) have shown tendencies toward politeness, formality, and implicit content moderation. While previous research has primarily focused on explicitly training models to moderate and detoxify sensitive content, there has been limited exploration of whether LLMs implicitly sanitize language without explicit instructions. This study empirically analyzes the implicit moderation behavior of GPT-4o-mini when paraphrasing sensitive content and evaluates the extent of sensitivity shifts. Our experiments indicate that GPT-4o-mini systematically moderates content toward less sensitive classes, with substantial reductions in derogatory and taboo language. Also, we evaluate the zero-shot capabilities of LLMs in classifying sentence sensitivity, comparing their performances against traditional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12365v2">Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      This survey paper outlines the key developments in the field of Large Language Models (LLMs), including enhancements to their reasoning skills, adaptability to various tasks, increased computational efficiency, and the ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. A significant focus is placed on efficiency, detailing scaling strategies, optimization techniques, and the influential Mixture-of-Experts (MoE) architecture, which strategically routes inputs to specialized subnetworks to boost predictive accuracy, while optimizing resource allocation. This survey also offers a broader perspective on recent advancements in LLMs, going beyond isolated aspects such as model architecture or ethical concerns. Additionally, it explores the role of LLMs in Agentic AI and their use as Autonomous Decision-Making Systems, and categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. The survey also identifies underexplored areas such as interpretability, cross-modal integration, and sustainability. While significant advancements have been made in LLMs, challenges such as high computational costs, biases, and ethical risks remain. Overcoming these requires a focus on bias mitigation, transparent decision-making, and explicit ethical guidelines. Future research will generally focus on enhancing the model's ability to handle multiple inputs, thereby making it more intelligent, safe, and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23261v1">DynaSwarm: Dynamically Graph Structure Selection for LLM-based Multi-agent System</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Current multi-agent systems (MAS) frameworks often rely on manually designed and static collaboration graph structures, limiting adaptability and performance. To address these limitations, we propose DynaSwarm, a dynamic framework that enhances LLM-based MAS through two key innovations: (1) an actor-critic reinforcement learning (A2C) mechanism to optimize graph structures with improved stability over prior RL methods, and (2) a dynamic graph selector that adaptively chooses the optimal graph structure for each input sample via parameter-efficient LLM fine-tuning. DynaSwarm eliminates the need for rigid, one-fits-all graph architectures, instead leveraging sample-specific idiosyncrasies to dynamically route queries through specialized agent networks. (c) We propose to fine-tune the demonstration retriever to fully exploit the power of in-context learning (ICL). Extensive experiments on question answering, mathematical reasoning, and coding tasks demonstrate that DynaSwarm consistently outperforms state-of-the-art single-agent and MAS baselines across multiple LLM backbones. Our findings highlight the importance of sample-aware structural flexibility in LLM MAS designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23248v1">Evaluating LLMs' Multilingual Capabilities for Bengali: Benchmark Creation and Performance Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Bengali is an underrepresented language in NLP research. However, it remains a challenge due to its unique linguistic structure and computational constraints. In this work, we systematically investigate the challenges that hinder Bengali NLP performance by focusing on the absence of standardized evaluation benchmarks. We then evaluated 10 recent open source Large Language Models (LLMs) in 8 of the translated datasets and performed a comprehensive error analysis to pinpoint their primary failure modes. Our findings reveal consistent performance gaps for Bengali compared to English, particularly for smaller models and specific model families like Mistral. We also identified promising robustness in certain architectures, such as DeepSeek, that maintain more stable performance across languages. Our analysis reveals an inverse relationship between tokenization efficiency and LLM accuracy where models tend to perform worse when inputs are excessively tokenized, whereas more efficient \& concise tokenization results in improved performance. These findings highlight critical areas where current models fall short and underscore the need for improved dataset quality and evaluation methodologies tailored to multilingual contexts. This work will catalyze further research on NLP for underrepresented languages, helping to democratize access to advanced language technologies worldwide. The code and dataset used in this research is publicly available at https://github.com/BengaliAI/bn-llm-benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15444v4">Cutting Through the Noise: Boosting LLM Performance on Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, PROBLEMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and improved ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to 6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00068v2">Framing Political Bias in Multilingual LLMs Across Pakistani Languages</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape public discourse, yet most evaluations of political and economic bias have focused on high-resource, Western languages and contexts. This leaves critical blind spots in low-resource, multilingual regions such as Pakistan, where linguistic identity is closely tied to political, religious, and regional ideologies. We present a systematic evaluation of political bias in 13 state-of-the-art LLMs across five Pakistani languages: Urdu, Punjabi, Sindhi, Pashto, and Balochi. Our framework integrates a culturally adapted Political Compass Test (PCT) with multi-level framing analysis, capturing both ideological stance (economic/social axes) and stylistic framing (content, tone, emphasis). Prompts are aligned with 11 socio-political themes specific to the Pakistani context. Results show that while LLMs predominantly reflect liberal-left orientations consistent with Western training data, they exhibit more authoritarian framing in regional languages, highlighting language-conditioned ideological modulation. We also identify consistent model-specific bias patterns across languages. These findings show the need for culturally grounded, multilingual bias auditing frameworks in global NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18666v3">AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Accepted by the 48th IEEE/ACM International Conference on Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identify 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23227v1">Enabling Few-Shot Alzheimer's Disease Diagnosis on Tabular Biomarker Data with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Early and accurate diagnosis of Alzheimer's disease (AD), a complex neurodegenerative disorder, requires analysis of heterogeneous biomarkers (e.g., neuroimaging, genetic risk factors, cognitive tests, and cerebrospinal fluid proteins) typically represented in a tabular format. With flexible few-shot reasoning, multimodal integration, and natural-language-based interpretability, large language models (LLMs) offer unprecedented opportunities for prediction with structured biomedical data. We propose a novel framework called TAP-GPT, Tabular Alzheimer's Prediction GPT, that adapts TableGPT2, a multimodal tabular-specialized LLM originally developed for business intelligence tasks, for AD diagnosis using structured biomarker data with small sample sizes. Our approach constructs few-shot tabular prompts using in-context learning examples from structured biomedical data and finetunes TableGPT2 using the parameter-efficient qLoRA adaption for a clinical binary classification task of AD or cognitively normal (CN). The TAP-GPT framework harnesses the powerful tabular understanding ability of TableGPT2 and the encoded prior knowledge of LLMs to outperform more advanced general-purpose LLMs and a tabular foundation model (TFM) developed for prediction tasks. To our knowledge, this is the first application of LLMs to the prediction task using tabular biomarker data, paving the way for future LLM-driven multi-agent frameworks in biomedical informatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15135v2">Controllable Traffic Simulation through LLM-Guided Hierarchical Reasoning and Refinement</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      Evaluating autonomous driving systems in complex and diverse traffic scenarios through controllable simulation is essential to ensure their safety and reliability. However, existing traffic simulation methods face challenges in their controllability. To address this, we propose a novel diffusion-based and LLM-enhanced traffic simulation framework. Our approach incorporates a high-level understanding module and a low-level refinement module, which systematically examines the hierarchical structure of traffic elements, guides LLMs to thoroughly analyze traffic scenario descriptions step by step, and refines the generation by self-reflection, enhancing their understanding of complex situations. Furthermore, we propose a Frenet-frame-based cost function framework that provides LLMs with geometrically meaningful quantities, improving their grasp of spatial relationships in a scenario and enabling more accurate cost function generation. Experiments on the Waymo Open Motion Dataset (WOMD) demonstrate that our method can handle more intricate descriptions and generate a broader range of scenarios in a controllable manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11122v2">Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Alpha factor mining is pivotal in quantitative investment for identifying predictive signals from complex financial data. While traditional formulaic alpha mining relies on human expertise, contemporary automated methods, such as those based on genetic programming or reinforcement learning, often struggle with search inefficiency or yield alpha factors that are difficult to interpret. This paper introduces a novel framework that integrates Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS) to overcome these limitations. Our framework leverages the LLM's instruction-following and reasoning capability to iteratively generate and refine symbolic alpha formulas within an MCTS-driven exploration. A key innovation is the guidance of MCTS exploration by rich, quantitative feedback from financial backtesting of each candidate factor, enabling efficient navigation of the vast search space. Furthermore, a frequent subtree avoidance mechanism is introduced to enhance search diversity and prevent formulaic homogenization, further improving performance. Experimental results on real-world stock market data demonstrate that our LLM-based framework outperforms existing methods by mining alphas with superior predictive accuracy and trading performance. The resulting formulas are also more amenable to human interpretation, establishing a more effective and efficient paradigm for formulaic alpha mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23209v1">Not Just What, But When: Integrating Irregular Intervals to LLM for Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Accepted by RecSys 2025 short paper track
    </div>
    <details class="paper-abstract">
      Time intervals between purchasing items are a crucial factor in sequential recommendation tasks, whereas existing approaches focus on item sequences and often overlook by assuming the intervals between items are static. However, dynamic intervals serve as a dimension that describes user profiling on not only the history within a user but also different users with the same item history. In this work, we propose IntervalLLM, a novel framework that integrates interval information into LLM and incorporates the novel interval-infused attention to jointly consider information of items and intervals. Furthermore, unlike prior studies that address the cold-start scenario only from the perspectives of users and items, we introduce a new viewpoint: the interval perspective to serve as an additional metric for evaluating recommendation methods on the warm and cold scenarios. Extensive experiments on 3 benchmarks with both traditional- and LLM-based baselines demonstrate that our IntervalLLM achieves not only 4.4% improvements in average but also the best-performing warm and cold scenarios across all users, items, and the proposed interval perspectives. In addition, we observe that the cold scenario from the interval perspective experiences the most significant performance drop among all recommendation methods. This finding underscores the necessity of further research on interval-based cold challenges and our integration of interval information in the realm of sequential recommendation tasks. Our code is available here: https://github.com/sony/ds-research-code/tree/master/recsys25-IntervalLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23170v1">BAR Conjecture: the Feasibility of Inference Budget-Constrained LLM Services with Authenticity and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      When designing LLM services, practitioners care about three key properties: inference-time budget, factual authenticity, and reasoning capacity. However, our analysis shows that no model can simultaneously optimize for all three. We formally prove this trade-off and propose a principled framework named The BAR Theorem for LLM-application design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02851v2">Leveraging LLMs to Create Content Corpora for Niche Domains</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 9 pages (main content), 5 figures. Supplementary materials can be found at https://github.com/pigfyy/30DayGen-Supplementary-Materials
    </div>
    <details class="paper-abstract">
      Constructing specialized content corpora from vast, unstructured web sources for domain-specific applications poses substantial data curation challenges. In this paper, we introduce a streamlined approach for generating high-quality, domain-specific corpora by efficiently acquiring, filtering, structuring, and cleaning web-based data. We showcase how Large Language Models (LLMs) can be leveraged to address complex data curation at scale, and propose a strategical framework incorporating LLM-enhanced techniques for structured content extraction and semantic deduplication. We validate our approach in the behavior education domain through its integration into 30 Day Me, a habit formation application. Our data pipeline, named 30DayGen, enabled the extraction and synthesis of 3,531 unique 30-day challenges from over 15K webpages. A user survey reports a satisfaction score of 4.3 out of 5, with 91% of respondents indicating willingness to use the curated content for their habit-formation goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23167v1">LENS: Learning Ensemble Confidence from Neural States for Multi-LLM Answer Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance across various tasks, with different models excelling in distinct domains and specific abilities. Effectively combining the predictions of multiple LLMs is crucial for enhancing system robustness and performance. However, existing ensemble methods often rely on simple techniques like voting or logits ensembling, which overlook the varying confidence and reliability of models in different contexts. In this work, we propose LENS (Learning ENsemble confidence from Neural States), a novel approach that learns to estimate model confidence by analyzing internal representations. For each LLM, we train a lightweight linear confidence predictor that leverages layer-wise hidden states and normalized probabilities as inputs. This allows for more nuanced weighting of model predictions based on their context-dependent reliability. Our method does not require modifying the model parameters and requires negligible additional computation. Experimental results on multiple-choice and boolean question-answering tasks demonstrate that LENS outperforms traditional ensemble methods by a substantial margin. Our findings suggest that internal representations provide valuable signals for determining model confidence and can be effectively leveraged for ensemble learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v1">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its inherently on-policy strategy with LLM's immense action space and sparse reward. Further, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel approach that synergizes internal exploitation (i.e., Thinking) with external data (i.e., Learning) to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components: Multiple Importance Sampling to address for distributional mismatch from external data, and an Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. The results show that RL-PLUS achieves state-of-the-art performance compared with existing RLVR methods on six math reasoning benchmarks and exhibits superior performance on six out-of-distribution reasoning tasks. It also achieves consistent and significant gains across diverse model families, with average relative improvements ranging from 21.1\% to 69.2\%. Moreover, Pass@k curves across multiple benchmarks indicate that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11878v2">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00217v1">Tabular Data Understanding with LLMs: A Survey of Recent Advances and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Tables have gained significant attention in large language models (LLMs) and multimodal large language models (MLLMs) due to their complex and flexible structure. Unlike linear text inputs, tables are two-dimensional, encompassing formats that range from well-structured database tables to complex, multi-layered spreadsheets, each with different purposes. This diversity in format and purpose has led to the development of specialized methods and tasks, instead of universal approaches, making navigation of table understanding tasks challenging. To address these challenges, this paper introduces key concepts through a taxonomy of tabular input representations and an introduction of table understanding tasks. We highlight several critical gaps in the field that indicate the need for further research: (1) the predominance of retrieval-focused tasks that require minimal reasoning beyond mathematical and logical operations; (2) significant challenges faced by models when processing complex table structures, large-scale tables, length context, or multi-table scenarios; and (3) the limited generalization of models across different tabular representations and formats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21760v2">MemInsight: Autonomous Memory Augmentation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have evolved to intelligently process information, make decisions, and interact with users or tools. A key capability is the integration of long-term memory capabilities, enabling these agents to draw upon historical interactions and knowledge. However, the growing memory size and need for semantic structuring pose significant challenges. In this work, we propose an autonomous memory augmentation approach, MemInsight, to enhance semantic data representation and retrieval mechanisms. By leveraging autonomous augmentation to historical interactions, LLM agents are shown to deliver more accurate and contextualized responses. We empirically validate the efficacy of our proposed approach in three task scenarios; conversational recommendation, question answering and event summarization. On the LLM-REDIAL dataset, MemInsight boosts persuasiveness of recommendations by up to 14%. Moreover, it outperforms a RAG baseline by 34% in recall for LoCoMo retrieval. Our empirical results show the potential of MemInsight to enhance the contextual performance of LLM agents across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00198v1">Testing the Untestable? An Empirical Study on the Testing Process of LLM-Powered Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Background: Software systems powered by large language models are becoming a routine part of everyday technologies, supporting applications across a wide range of domains. In software engineering, many studies have focused on how LLMs support tasks such as code generation, debugging, and documentation. However, there has been limited focus on how full systems that integrate LLMs are tested during development. Aims: This study explores how LLM-powered systems are tested in the context of real-world application development. Method: We conducted an exploratory case study using 99 individual reports written by students who built and deployed LLM-powered applications as part of a university course. Each report was independently analyzed using thematic analysis, supported by a structured coding process. Results: Testing strategies combined manual and automated methods to evaluate both system logic and model behavior. Common practices included exploratory testing, unit testing, and prompt iteration. Reported challenges included integration failures, unpredictable outputs, prompt sensitivity, hallucinations, and uncertainty about correctness. Conclusions: Testing LLM-powered systems required adaptations to traditional verification methods, blending source-level reasoning with behavior-aware evaluations. These findings provide evidence on the practical context of testing generative components in software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00161v1">Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution. In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision. For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation. Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06509v2">Private GPTs for LLM-driven testing in software development and machine learning</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 5 pages, 10 figures
    </div>
    <details class="paper-abstract">
      In this contribution, we examine the capability of private GPTs to automatically generate executable test code based on requirements. More specifically, we use acceptance criteria as input, formulated as part of epics, or stories, which are typically used in modern development processes. This gives product owners, or business intelligence, respectively, a way to directly produce testable criteria through the use of LLMs. We explore the quality of the so-produced tests in two ways: i) directly by letting the LLM generate code from requirements, ii) through an intermediate step using Gherkin syntax. As a result, it turns out that the two-step procedure yields better results -where we define better in terms of human readability and best coding practices, i.e. lines of code and use of additional libraries typically used in testing. Concretely, we evaluate prompt effectiveness across two scenarios: a simple "Hello World" program and a digit classification model, showing that structured prompts lead to higher-quality test outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00086v1">Do LLMs produce texts with "human-like" lexical diversity?</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 35 pages; includes abstract
    </div>
    <details class="paper-abstract">
      The degree to which LLMs produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (-3.5, -4, -o4 mini, and -4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAS, and Support Vector Machines revealed that the LLM-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and -4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that LLMs do not produce human-like texts in relation to lexical diversity, and the newer LLMs produce less human-like texts than older models. We discuss the implications of these results for language pedagogy and related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00083v1">A Survey on Code Generation with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21170v2">OneShield -- the Next Generation of LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models has created a general excitement about the great potential for a myriad of applications. While LLMs offer many possibilities, questions about safety, privacy, and ethics have emerged, and all the key actors are working to address these issues with protective measures for their own models and standalone solutions. The constantly evolving nature of LLMs makes it extremely challenging to universally shield users against their potential risks, and one-size-fits-all solutions are unfeasible. In this work, we propose OneShield, our stand-alone, model-agnostic and customizable solution to safeguard LLMs. OneShield aims to provide facilities for defining risk factors, expressing and declaring contextual safety and compliance policies, and mitigating LLM risks, with a focus on each specific customer. We describe the implementation of the framework, discuss scalability considerations, and provide usage statistics of OneShield since its initial deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00933v1">OKG-LLM: Aligning Ocean Knowledge Graph with Observation Data via LLMs for Global Sea Surface Temperature Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Sea surface temperature (SST) prediction is a critical task in ocean science, supporting various applications, such as weather forecasting, fisheries management, and storm tracking. While existing data-driven methods have demonstrated significant success, they often neglect to leverage the rich domain knowledge accumulated over the past decades, limiting further advancements in prediction accuracy. The recent emergence of large language models (LLMs) has highlighted the potential of integrating domain knowledge for downstream tasks. However, the application of LLMs to SST prediction remains underexplored, primarily due to the challenge of integrating ocean domain knowledge and numerical data. To address this issue, we propose Ocean Knowledge Graph-enhanced LLM (OKG-LLM), a novel framework for global SST prediction. To the best of our knowledge, this work presents the first systematic effort to construct an Ocean Knowledge Graph (OKG) specifically designed to represent diverse ocean knowledge for SST prediction. We then develop a graph embedding network to learn the comprehensive semantic and structural knowledge within the OKG, capturing both the unique characteristics of individual sea regions and the complex correlations between them. Finally, we align and fuse the learned knowledge with fine-grained numerical SST data and leverage a pre-trained LLM to model SST patterns for accurate prediction. Extensive experiments on the real-world dataset demonstrate that OKG-LLM consistently outperforms state-of-the-art methods, showcasing its effectiveness, robustness, and potential to advance SST prediction. The codes are available in the online repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08450v2">IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19073v2">MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanations</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22050v2">DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 22 pages, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. Our codes are available at https://github.com/MinghoKwok/DeepSieve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22758v1">MASCA: LLM based-Multi Agents System for Credit Assessment</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted at ACL REALM Workshop. Work in Progress
    </div>
    <details class="paper-abstract">
      Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22753v1">Opportunities and Challenges of LLMs in Education: An NLP Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Interest in the role of large language models (LLMs) in education is increasing, considering the new opportunities they offer for teaching, learning, and assessment. In this paper, we examine the impact of LLMs on educational NLP in the context of two main application scenarios: {\em assistance} and {\em assessment}, grounding them along the four dimensions -- reading, writing, speaking, and tutoring. We then present the new directions enabled by LLMs, and the key challenges to address. We envision that this holistic overview would be useful for NLP researchers and practitioners interested in exploring the role of LLMs in developing language-focused and NLP-enabled educational applications of the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22716v1">From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: https://github.com/probe2/TIRESRAG-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22619v1">Enhancing Manufacturing Knowledge Access with LLMs and Context-aware Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 European Conference on Artificial Intelligence (ECAI) 2024
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs) have transformed data management within the manufacturing industry, offering effective means for integrating disparate data sources through shared and structured conceptual schemas. However, harnessing the power of KGs can be daunting for non-experts, as it often requires formulating complex SPARQL queries to retrieve specific information. With the advent of Large Language Models (LLMs), there is a growing potential to automatically translate natural language queries into the SPARQL format, thus bridging the gap between user-friendly interfaces and the sophisticated architecture of KGs. The challenge remains in adequately informing LLMs about the relevant context and structure of domain-specific KGs, e.g., in manufacturing, to improve the accuracy of generated queries. In this paper, we evaluate multiple strategies that use LLMs as mediators to facilitate information retrieval from KGs. We focus on the manufacturing domain, particularly on the Bosch Line Information System KG and the I40 Core Information Model. In our evaluation, we compare various approaches for feeding relevant context from the KG to the LLM and analyze their proficiency in transforming real-world questions into SPARQL queries. Our findings show that LLMs can significantly improve their performance on generating correct and complete queries when provided only the adequate context of the KG schema. Such context-aware prompting techniques help LLMs to focus on the relevant parts of the ontology and reduce the risk of hallucination. We anticipate that the proposed techniques help LLMs to democratize access to complex data repositories and empower informed decision-making in manufacturing settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22565v1">Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22564v1">Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22467v1">Towards Simulating Social Influence Dynamics with LLM-based Multi-agents</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models offer promising capabilities to simulate complex human social interactions. We investigate whether LLM-based multi-agent simulations can reproduce core human social dynamics observed in online forums. We evaluate conformity dynamics, group polarization, and fragmentation across different model scales and reasoning capabilities using a structured simulation framework. Our findings indicate that smaller models exhibit higher conformity rates, whereas models optimized for reasoning are more resistant to social influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09213v3">FineMedLM-o1: Enhancing Medical Knowledge Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the deep reasoning required for complex medical problems, such as differential diagnosis and medication recommendations. We propose FineMedLM-o1, which leverages high-quality medical synthetic data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduce Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also propose a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22447v1">Breaking Obfuscation: Cluster-Aware Graph with LLM-Aided Recovery for Malicious JavaScript Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      With the rapid expansion of web-based applications and cloud services, malicious JavaScript code continues to pose significant threats to user privacy, system integrity, and enterprise security. But, detecting such threats remains challenging due to sophisticated code obfuscation techniques and JavaScript's inherent language characteristics, particularly its nested closure structures and syntactic flexibility. In this work, we propose DeCoda, a hybrid defense framework that combines large language model (LLM)-based deobfuscation with code graph learning: (1) We first construct a sophisticated prompt-learning pipeline with multi-stage refinement, where the LLM progressively reconstructs the original code structure from obfuscated inputs and then generates normalized Abstract Syntax Tree (AST) representations; (2) In JavaScript ASTs, dynamic typing scatters semantically similar nodes while deeply nested functions fracture scope capturing, introducing structural noise and semantic ambiguity. To address these challenges, we then propose to learn hierarchical code graph representations via a Cluster-wise Graph that synergistically integrates graph transformer network, node clustering, and node-to-cluster attention to simultaneously capture both local node-level semantics and global cluster-induced structural relationships from AST graph. Experimental results demonstrate that our method achieves F1-scores of 94.64% and 97.71% on two benchmark datasets, demonstrating absolute improvements of 10.74% and 13.85% over state-of-the-art baselines. In false-positive control evaluation at fixed FPR levels (0.0001, 0.001, 0.01), our approach delivers 4.82, 5.91, and 2.53 higher TPR respectively compared to the best-performing baseline. These results highlight the effectiveness of LLM-based deobfuscation and underscore the importance of modeling cluster-level relationships in detecting malicious code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15790v3">ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 4 pages, 1 figure. To appear in Proceedings of the 16th Asia-Pacific Symposium on Internetware (Internetware 2025), ACM ICPS. DOI: https://doi.org/10.1145/3755881.3755934
    </div>
    <details class="paper-abstract">
      With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21990v2">ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 13 figures, 4 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoner LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized knowledge points to enhance the model's understanding of the fundamental principles and logical structure of chemistry. Then, we propose a mix-sourced distillation strategy that integrates expert-curated knowledge with general-domain reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the reliability, transparency, and practical utility of the model in real-world human-AI collaboration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22387v1">PATENTWRITER: A Benchmarking Study for Patent Drafting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as transformative approaches in several important fields. This paper aims for a paradigm shift for patent writing by leveraging LLMs to overcome the tedious patent-filing process. In this work, we present PATENTWRITER, the first unified benchmarking framework for evaluating LLMs in patent abstract generation. Given the first claim of a patent, we evaluate six leading LLMs -- including GPT-4 and LLaMA-3 -- under a consistent setup spanning zero-shot, few-shot, and chain-of-thought prompting strategies to generate the abstract of the patent. Our benchmark PATENTWRITER goes beyond surface-level evaluation: we systematically assess the output quality using a comprehensive suite of metrics -- standard NLP measures (e.g., BLEU, ROUGE, BERTScore), robustness under three types of input perturbations, and applicability in two downstream patent classification and retrieval tasks. We also conduct stylistic analysis to assess length, readability, and tone. Experimental results show that modern LLMs can generate high-fidelity and stylistically appropriate patent abstracts, often surpassing domain-specific baselines. Our code and dataset are open-sourced to support reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22367v1">Traits Run Deep: Enhancing Personality Assessment via Psychology-Guided LLM Representations and Multimodal Apparent Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 8 pages, 3 figures, ACM MM 2025
    </div>
    <details class="paper-abstract">
      Accurate and reliable personality assessment plays a vital role in many fields, such as emotional intelligence, mental health diagnostics, and personalized education. Unlike fleeting emotions, personality traits are stable, often subconsciously leaked through language, facial expressions, and body behaviors, with asynchronous patterns across modalities. It was hard to model personality semantics with traditional superficial features and seemed impossible to achieve effective cross-modal understanding. To address these challenges, we propose a novel personality assessment framework called \textit{\textbf{Traits Run Deep}}. It employs \textit{\textbf{psychology-informed prompts}} to elicit high-level personality-relevant semantic representations. Besides, it devises a \textit{\textbf{Text-Centric Trait Fusion Network}} that anchors rich text semantics to align and integrate asynchronous signals from other modalities. To be specific, such fusion module includes a Chunk-Wise Projector to decrease dimensionality, a Cross-Modal Connector and a Text Feature Enhancer for effective modality fusion and an ensemble regression head to improve generalization in data-scarce situations. To our knowledge, we are the first to apply personality-specific prompts to guide large language models (LLMs) in extracting personality-aware semantics for improved representation quality. Furthermore, extracting and fusing audio-visual apparent behavior features further improves the accuracy. Experimental results on the AVI validation set have demonstrated the effectiveness of the proposed components, i.e., approximately a 45\% reduction in mean squared error (MSE). Final evaluations on the test set of the AVI Challenge 2025 confirm our method's superiority, ranking first in the Personality Assessment track. The source code will be made available at https://github.com/MSA-LMC/TraitsRunDeep.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20300v2">Talking-to-Build: How LLM-Assisted Interface Shapes Player Performance and Experience in Minecraft</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      With large language models (LLMs) on the rise, in-game interactions are shifting from rigid commands to natural conversations. However, the impacts of LLMs on player performance and game experience remain underexplored. This work explores LLM's role as a co-builder during gameplay, examining its impact on task performance, usability, and player experience. Using Minecraft as a sandbox, we present an LLM-assisted interface that engages players through natural language, aiming to facilitate creativity and simplify complex gaming commands. We conducted a mixed-methods study with 30 participants, comparing LLM-assisted and command-based interfaces across simple and complex game tasks. Quantitative and qualitative analyses reveal that the LLM-assisted interface significantly improves player performance, engagement, and overall game experience. Additionally, task complexity has a notable effect on player performance and experience across both interfaces. Our findings highlight the potential of LLM-assisted interfaces to revolutionize virtual experiences, emphasizing the importance of balancing intuitiveness with predictability, transparency, and user agency in AI-driven, multimodal gaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22352v1">Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 15 pages, 8 figures. Published at the 7th ACM Conference on Conversational User Interfaces (CUI '25), July 8-10, 2025, Waterloo, Canada. Open-source code available at https://github.com/ISUE/iva-cui
    </div>
    <details class="paper-abstract">
      We investigated the challenges of mitigating response delays in free-form conversations with virtual agents powered by Large Language Models (LLMs) within Virtual Reality (VR). For this, we used conversational fillers, such as gestures and verbal cues, to bridge delays between user input and system responses and evaluate their effectiveness across various latency levels and interaction scenarios. We found that latency above 4 seconds degrades quality of experience, while natural conversational fillers improve perceived response time, especially in high-delay conditions. Our findings provide insights for practitioners and researchers to optimize user engagement whenever conversational systems' responses are delayed by network limitations or slow hardware. We also contribute an open-source pipeline that streamlines deploying conversational agents in virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01282v2">Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 accepted in COLM2025, 9 pages
    </div>
    <details class="paper-abstract">
      While the inconsistency of LLMs is not a novel topic, prior research has predominantly addressed two types of generative inconsistencies: i) Randomness Inconsistency: running the same LLM multiple trials, yielding varying responses; ii) Paraphrase Inconsistency: paraphrased prompts result in different responses from the same LLM. Randomness Inconsistency arises from the inherent randomness due to stochastic sampling in generative models, while Paraphrase Inconsistency is a consequence of the language modeling objectives, where paraphrased prompts alter the distribution of vocabulary logits. This research discovers Prompt-Reverse Inconsistency (PRIN), a new form of LLM self-inconsistency: given a question and a couple of LLM-generated answer candidates, the LLM often has conflicting responses when prompted "Which are correct answers?" and "Which are incorrect answers?". PRIN poses a big concern as it undermines the credibility of LLM-as-a-judge, and suggests a challenge for LLMs to adhere to basic logical rules. We conduct a series of experiments to investigate PRIN, examining the extent of PRIN across different LLMs, methods to mitigate it, potential applications, and its relationship with Randomness Inconsistency and Paraphrase Inconsistency. As the first study to explore PRIN, our findings offer valuable insights into the inner workings of LLMs and contribute to advancing trustworthy AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22326v1">An Explainable Emotion Alignment Framework for LLM-Empowered Agent in Metaverse Service Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Metaverse service is a product of the convergence between Metaverse and service systems, designed to address service-related challenges concerning digital avatars, digital twins, and digital natives within Metaverse. With the rise of large language models (LLMs), agents now play a pivotal role in Metaverse service ecosystem, serving dual functions: as digital avatars representing users in the virtual realm and as service assistants (or NPCs) providing personalized support. However, during the modeling of Metaverse service ecosystems, existing LLM-based agents face significant challenges in bridging virtual-world services with real-world services, particularly regarding issues such as character data fusion, character knowledge association, and ethical safety concerns. This paper proposes an explainable emotion alignment framework for LLM-based agents in Metaverse Service Ecosystem. It aims to integrate factual factors into the decision-making loop of LLM-based agents, systematically demonstrating how to achieve more relational fact alignment for these agents. Finally, a simulation experiment in the Offline-to-Offline food delivery scenario is conducted to evaluate the effectiveness of this framework, obtaining more realistic social emergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19747v2">TokenBlowUp: Resolving Representational Singularities in LLM Token Spaces via Monoidal Transformations</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Recent work has provided compelling evidence challenging the foundational manifold hypothesis for the token embedding spaces of Large Language Models (LLMs). These findings reveal the presence of geometric singularities around polysemous tokens, which can lead to representational instability. Existing methodologies, which presuppose a smooth data manifold, are ill-equipped to address such intrinsic structural flaws. In this paper, we formalize this problem in the language of scheme theory and propose a rigorous resolution by applying the scheme-theoretic blow-up at each singular point. This procedure replaces a singular point in the ambient affine scheme with its exceptional divisor, which we identify as a canonical geometric space -- a projective space of directions -- that houses the disambiguated semantic meanings of the token. This process of ``representational desingularization'' constructs a new geometric landscape for embeddings. We prove a formal theorem guaranteeing the geometric regularization of this new space, showing that the original pathologies are resolved. Finally, we outline the architectural implications of our framework, arguing for a paradigm shift from static look-ups to dynamic, geometrically-grounded computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23158v1">User Feedback in Human-LLM Dialogues: A Lens to Understand Users But Noisy as a Learning Signal</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Earlier version of this paper was presented at 2nd Workshop on Models of Human Feedback for AI Alignment (MoFA), ICML 2025
    </div>
    <details class="paper-abstract">
      Once language models (LMs) are deployed, they can interact with users long-term, ideally evolving continuously based on their feedback. Asking for direct user feedback can be disruptive; thus, we study harvesting user feedback from user-LM interaction logs. We study implicit user feedback in two user-LM interaction datasets (WildChat and LMSYS). First, we analyze user feedback in the user-LLM conversation trajectory, providing insights into when and why such feedback occurs. Second, we study harvesting learning signals from such implicit user feedback. We find that the contents of user feedback (e.g., user wanted clarification), not just the polarity (e.g., users were unhappy with the previous model response), can improve model performance in short human-designed questions (MTBench) but not on longer and more complex questions (WildBench). We also find that the usefulness of user feedback is largely tied to the quality of the user's initial prompt. Together, we provide an in-depth study of implicit user feedback, showing its potential and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23121v1">Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted at KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models (Agentic & GenAI Evaluation Workshop KDD '25)
    </div>
    <details class="paper-abstract">
      In this work, we study a critical research problem regarding the trustworthiness of large language models (LLMs): how LLMs behave when encountering ambiguous narrative text, with a particular focus on Chinese textual ambiguity. We created a benchmark dataset by collecting and generating ambiguous sentences with context and their corresponding disambiguated pairs, representing multiple possible interpretations. These annotated examples are systematically categorized into 3 main categories and 9 subcategories. Through experiments, we discovered significant fragility in LLMs when handling ambiguity, revealing behavior that differs substantially from humans. Specifically, LLMs cannot reliably distinguish ambiguous text from unambiguous text, show overconfidence in interpreting ambiguous text as having a single meaning rather than multiple meanings, and exhibit overthinking when attempting to understand the various possible meanings. Our findings highlight a fundamental limitation in current LLMs that has significant implications for their deployment in real-world applications where linguistic ambiguity is common, calling for improved approaches to handle uncertainty in language understanding. The dataset and code are publicly available at this GitHub repository: https://github.com/ictup/LLM-Chinese-Textual-Disambiguation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23087v1">On LLM-Assisted Generation of Smart Contracts from Business Processes</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted at the Workshop on Distributed Ledger Technologies in Business Process Management, At the International Conference for Business Process Management (BPM), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have changed the reality of how software is produced. Within the wider software engineering community, among many other purposes, they are explored for code generation use cases from different types of input. In this work, we present an exploratory study to investigate the use of LLMs for generating smart contract code from business process descriptions, an idea that has emerged in recent literature to overcome the limitations of traditional rule-based code generation approaches. However, current LLM-based work evaluates generated code on small samples, relying on manual inspection, or testing whether code compiles but ignoring correct execution. With this work, we introduce an automated evaluation framework and provide empirical data from larger data sets of process models. We test LLMs of different types and sizes in their capabilities of achieving important properties of process execution, including enforcing process flow, resource allocation, and data-based conditions. Our results show that LLM performance falls short of the perfect reliability required for smart contract development. We suggest future work to explore responsible LLM integrations in existing tools for code generation to ensure more reliable output. Our benchmarking framework can serve as a foundation for developing and evaluating such integrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23035v1">KLLM: Fast LLM Inference with K-Means Quantization</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference poses significant challenges due to its intensive memory and computation demands. Weight and activation quantization (WAQ) offers a promising solution by reducing both memory footprint and arithmetic complexity. However, two key challenges remain in the existing WAQ designs. (1) Traditional WAQ designs rely on uniform integer-based quantization for hardware efficiency, but this often results in significant accuracy degradation at low precision. K-Means-based quantization, a non-uniform quantization technique, achieves higher accuracy by matching the Gaussian-like distributions of weights and activations in LLMs. However, its non-uniform nature prevents direct execution on low-precision compute units, requiring dequantization and floating-point matrix multiplications (MatMuls) during inference. (2) Activation outliers further hinder effective low-precision WAQ. Offline thresholding methods for outlier detection can lead to significant model performance degradation, while existing online detection techniques introduce substantial runtime overhead. To address the aforementioned challenges and fully unleash the potential of WAQ with K-Means quantization for LLM inference, in this paper, we propose KLLM, a hardware-software co-design framework. KLLM features an index-based computation scheme for efficient execution of MatMuls and nonlinear operations on K-Means-quantized data, which avoids most of the dequantization and full-precision computations. Moreover, KLLM incorporates a novel outlier detection engine, Orizuru, that efficiently identifies the top-$k$ largest and smallest elements in the activation data stream during online inference. Extensive experiments show that, on average, KLLM achieves speedups of 9.67x, 7.03x and energy efficiency improvements of 229.50x, 150.21x compared to the A100 GPU and Atom, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00032v1">Strategic Communication and Language Bias in Multi-Agent LLM Coordination</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents are increasingly deployed in multi-agent scenarios where coordination is crucial but not always assured. Previous studies indicate that the language used to frame strategic scenarios can influence cooperative behavior. This paper explores whether allowing agents to communicate amplifies these language-driven effects. Leveraging the FAIRGAME framework, we simulate one-shot and repeated games across different languages and models, both with and without communication. Our experiments, conducted with two advanced LLMs, GPT-4o and Llama 4 Maverick, reveal that communication significantly influences agent behavior, though its impact varies by language, personality, and game structure. These findings underscore the dual role of communication in fostering coordination and reinforcing biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00031v1">Git Context Controller: Manage the Context of LLM-based Agents like Git</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 in updating
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based agents have shown impressive capabilities by interleaving internal reasoning with external tool use. However, as these agents are deployed in long-horizon workflows, such as coding for a big, long-term project, context management becomes a critical bottleneck. We introduce Git-Context-Controller (GCC), a structured context management framework inspired by software version control systems. GCC elevates context as versioned memory hierarchy like Git. It structures agent memory as a persistent file system with explicit operations: COMMIT, BRANCH, MERGE, and CONTEXT, enabling milestone-based checkpointing, exploration of alternative plans, and structured reflection. Our approach empowers agents to manage long-term goals, isolate architectural experiments, and recover or hand off memory across sessions and agents. Empirically, agents equipped with GCC achieve state-of-the-art performance on the SWE-Bench-Lite benchmark, resolving 48.00 of software bugs, outperforming 26 competitive systems. In a self-replication case study, a GCC-augmented agent builds a new CLI agent from scratch, achieving 40.7 task resolution, compared to only 11.7 without GCC. The code is released at: https://github.com/theworldofagents/GCC
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15549v3">Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Code at https://github.com/aengusl/latent-adversarial-training. Models at https://huggingface.co/LLM-LAT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07237v2">Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance across a range of natural language processing tasks. However, their vast number of parameters introduces significant memory challenges during training, particularly when using memory-intensive optimizers like Adam. Existing memory-efficient algorithms often rely on techniques such as singular value decomposition projection or weight freezing. While these approaches help alleviate memory constraints, they generally produce suboptimal results compared to full-rank updates. In this paper, we investigate the memory-efficient method beyond low-rank training, proposing a novel solution called Gradient Wavelet Transform (GWT), which applies wavelet transforms to gradients in order to significantly reduce the memory requirements for maintaining optimizer states. We demonstrate that GWT can be seamlessly integrated with memory-intensive optimizers, enabling efficient training without sacrificing performance. Through extensive experiments on both pre-training and fine-tuning tasks, we show that GWT achieves state-of-the-art performance compared with advanced memory-efficient optimizers and full-rank approaches in terms of both memory usage and training performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03248v2">AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that our method substantially reduces computation load (e.g., a $\textbf{7-fold}$ reduction in FLOPs) while preserving the performance of video and image LLMs. Further, at a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., $\textbf{+4.6}$ on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code is available at https://github.com/LaVi-Lab/AIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21563v1">Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Recommendation systems often suffer from data sparsity caused by limited user-item interactions, which degrade their performance and amplify popularity bias in real-world scenarios. This paper proposes a novel data augmentation framework that leverages Large Language Models (LLMs) and item textual descriptions to enrich interaction data. By few-shot prompting LLMs multiple times to rerank items and aggregating the results via majority voting, we generate high-confidence synthetic user-item interactions, supported by theoretical guarantees based on the concentration of measure. To effectively leverage the augmented data in the context of a graph recommendation system, we integrate it into a graph contrastive learning framework to mitigate distributional shift and alleviate popularity bias. Extensive experiments show that our method improves accuracy and reduces popularity bias, outperforming strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20511v2">Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19537v2">Mind the Language Gap in Digital Humanities: LLM-Aided Translation of SKOS Thesauri</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      We introduce WOKIE, an open-source, modular, and ready-to-use pipeline for the automated translation of SKOS thesauri. This work addresses a critical need in the Digital Humanities (DH), where language diversity can limit access, reuse, and semantic interoperability of knowledge resources. WOKIE combines external translation services with targeted refinement using Large Language Models (LLMs), balancing translation quality, scalability, and cost. Designed to run on everyday hardware and be easily extended, the application requires no prior expertise in machine translation or LLMs. We evaluate WOKIE across several DH thesauri in 15 languages with different parameters, translation services and LLMs, systematically analysing translation quality, performance, and ontology matching improvements. Our results show that WOKIE is suitable to enhance the accessibility, reuse, and cross-lingual interoperability of thesauri by hurdle-free automated translation and improved ontology matching performance, supporting more inclusive and multilingual research infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21538v1">Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21507v1">VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 21 pages, 19 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Video Anomaly Detection (VAD) aims to identify anomalous events in videos and accurately determine their time intervals. Current VAD methods mainly fall into two categories: traditional DNN-based approaches that focus on temporal localization, and LLM-based approaches that emphasize semantic understanding. Both anomaly understanding and grounding are essential for comprehensive video anomaly detection and can complement each other. However, no existing model or dataset supports both tasks simultaneously. To address this, we introduce VAGU (Video Anomaly Grounding and Understanding), the first benchmark to integrate both tasks. Each VAGU instance includes annotations for anomaly category, semantic explanation, precise temporal grounding and Video QA. We also provide multiple-choice Video QA for objective evaluation. Based on this dataset, we propose Glance then Scrutinize (GtS), a training-free framework guided by textual prompts. The framework first enables coarse localization of high-probability anomalous regions, followed by detailed anomaly interpretation and temporal boundary refinement. Additionally, we propose the JeAUG metric, which jointly evaluates semantic interpretability and temporal precision, overcoming the limitations of traditional metrics. Extensive experiments verify the effectiveness of our benchmark, framework, and evaluation metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14479v4">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21504v1">Evaluation and Benchmarking of LLM Agents: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The rise of LLM-based agents has opened new frontiers in AI applications, yet evaluating these agents remains a complex and underdeveloped area. This survey provides an in-depth overview of the emerging field of LLM agent evaluation, introducing a two-dimensional taxonomy that organizes existing work along (1) evaluation objectives -- what to evaluate, such as agent behavior, capabilities, reliability, and safety -- and (2) evaluation process -- how to evaluate, including interaction modes, datasets and benchmarks, metric computation methods, and tooling. In addition to taxonomy, we highlight enterprise-specific challenges, such as role-based access to data, the need for reliability guarantees, dynamic and long-horizon interactions, and compliance, which are often overlooked in current research. We also identify future research directions, including holistic, more realistic, and scalable evaluation. This work aims to bring clarity to the fragmented landscape of agent evaluation and provide a framework for systematic assessment, enabling researchers and practitioners to evaluate LLM agents for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21485v1">HLSDebugger: Identification and Correction of Logic Bugs in HLS Code with LLM Solutions</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 This work has been accepted at ICCAD 2025 (International Conference on Computer-Aided Design)
    </div>
    <details class="paper-abstract">
      High-level synthesis (HLS) accelerates hardware design by enabling the automatic translation of high-level descriptions into efficient hardware implementations. However, debugging HLS code is a challenging and labor-intensive task, especially for novice circuit designers or software engineers without sufficient hardware domain knowledge. The recent emergence of Large Language Models (LLMs) is promising in automating the HLS debugging process. Despite the great potential, three key challenges persist when applying LLMs to HLS logic debugging: 1) High-quality circuit data for training LLMs is scarce, posing a significant challenge. 2) Debugging logic bugs in hardware is inherently more complex than identifying software bugs with existing golden test cases. 3) The absence of reliable test cases requires multi-tasking solutions, performing both bug identification and correction. complicates the multi-tasking required for effective HLS debugging. In this work, we propose a customized solution named HLSDebugger to address the challenges. HLSDebugger first generates and releases a large labeled dataset with 300K data samples, targeting HLS logic bugs. The HLSDebugger model adopts an encoder-decoder structure, performing bug location identification, bug type prediction, and bug correction with the same model. HLSDebugger significantly outperforms advanced LLMs like GPT-4 in bug identification and by more than 3x in bug correction. It makes a substantial advancement in the exploration of automated debugging of HLS code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21482v1">Improving Task Diversity in Label Efficient Supervised Finetuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but developing high-performing models for specialized applications often requires substantial human annotation -- a process that is time-consuming, labor-intensive, and expensive. In this paper, we address the label-efficient learning problem for supervised finetuning (SFT) by leveraging task-diversity as a fundamental principle for effective data selection. This is markedly different from existing methods based on the prompt-diversity. Our approach is based on two key observations: 1) task labels for different prompts are often readily available; 2) pre-trained models have significantly varying levels of confidence across tasks. We combine these facts to devise a simple yet effective sampling strategy: we select examples across tasks using an inverse confidence weighting strategy. This produces models comparable to or better than those trained with more complex sampling procedures, while being significantly easier to implement and less computationally intensive. Notably, our experimental results demonstrate that this method can achieve better accuracy than training on the complete dataset (a 4\% increase in MMLU score). Across various annotation budgets and two instruction finetuning datasets, our algorithm consistently performs at or above the level of the best existing methods, while reducing annotation costs by up to 80\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21476v1">Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      We present HumorBench, a benchmark designed to evaluate large language models' (LLMs) ability to reason about and explain sophisticated humor in cartoon captions. As reasoning models increasingly saturate existing benchmarks in mathematics and science, novel and challenging evaluations of model intelligence beyond STEM domains are essential. Reasoning is fundamentally involved in text-based humor comprehension, requiring the identification of connections between concepts in cartoons/captions and external cultural references, wordplays, and other mechanisms. HumorBench includes approximately 300 unique cartoon-caption pairs from the New Yorker Caption Contest and Cartoonstock.com, with expert-annotated evaluation rubrics identifying essential joke elements. LLMs are evaluated based on their explanations towards the humor and abilities in identifying the joke elements. To perform well on this task, models must form and test hypotheses about associations between concepts, potentially backtracking from initial interpretations to arrive at the most plausible explanation. Our extensive benchmarking of current SOTA models reveals three key insights: (1) LLM progress on STEM reasoning transfers effectively to humor comprehension; (2) models trained exclusively on STEM reasoning data still perform well on HumorBench, demonstrating strong transferability of reasoning abilities; and (3) test-time scaling by increasing thinking token budgets yields mixed results across different models in humor reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21471v1">An LLM Driven Agent Framework for Automated Infrared Spectral Multi Task Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Infrared spectroscopy offers rapid, non destructive measurement of chemical and material properties but suffers from high dimensional, overlapping spectral bands that challenge conventional chemometric approaches. Emerging large language models (LLMs), with their capacity for generalization and reasoning, offer promising potential for automating complex scientific workflows. Despite this promise, their application in IR spectral analysis remains largely unexplored. This study addresses the critical challenge of achieving accurate, automated infrared spectral interpretation under low-data conditions using an LLM-driven framework. We introduce an end-to-end, large language model driven agent framework that integrates a structured literature knowledge base, automated spectral preprocessing, feature extraction, and multi task reasoning in a unified pipeline. By querying a curated corpus of peer reviewed IR publications, the agent selects scientifically validated routines. The selected methods transform each spectrum into low dimensional feature sets, which are fed into few shot prompt templates for classification, regression, and anomaly detection. A closed loop, multi turn protocol iteratively appends mispredicted samples to the prompt, enabling dynamic refinement of predictions. Across diverse materials: stamp pad ink, Chinese medicine, Pu'er tea, Citri Reticulatae Pericarpium and waste water COD datasets, the multi turn LLM consistently outperforms single turn inference, rivaling or exceeding machine learning and deep learning models under low data regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16491v3">BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in language. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21447v1">LLM4VV: Evaluating Cutting-Edge LLMs for Generation and Evaluation of Directive-Based Parallel Programming Model Compiler Tests</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The usage of Large Language Models (LLMs) for software and test development has continued to increase since LLMs were first introduced, but only recently have the expectations of LLMs become more realistic. Verifying the correctness of code generated by LLMs is key to improving their usefulness, but there have been no comprehensive and fully autonomous solutions developed yet. Hallucinations are a major concern when LLMs are applied blindly to problems without taking the time and effort to verify their outputs, and an inability to explain the logical reasoning of LLMs leads to issues with trusting their results. To address these challenges while also aiming to effectively apply LLMs, this paper proposes a dual-LLM system (i.e. a generative LLM and a discriminative LLM) and experiments with the usage of LLMs for the generation of a large volume of compiler tests. We experimented with a number of LLMs possessing varying parameter counts and presented results using ten carefully-chosen metrics that we describe in detail in our narrative. Through our findings, it is evident that LLMs possess the promising potential to generate quality compiler tests and verify them automatically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19855v2">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v5">InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 take a middle-ground approach, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfiniteHBD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfiniteHBD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt to variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh)-based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfiniteHBD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios are under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12225v2">Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Enhancing Large Language Model (LLM)'s performance with best-of-N sampling is effective and has attracted significant attention. However, it is computationally prohibitive due to massive, data-hungry text-based reward models. By changing the data source from text to hidden states, we introduce SWIFT (Simple Weighted Intrinsic Feedback Technique), a novel, lightweight technique that leverages the rich information embedded in LLM hidden states to address these issues, which operates on token-level and consists of only linear layers. Extensive experiments show that SWIFT outperforms baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training, demonstrating significant efficiency improvement. SWIFT's robust scalability, applicability to some closed-source models via logits, and ability to be combined with traditional reward models to yield further performance gains underscore its practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21428v1">MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agent Multi-Turn Conversations</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 23 Pages, 20 Figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have shown significant autonomous capabilities in dynamically searching and incorporating relevant tools or Model Context Protocol (MCP) servers for individual queries. However, fixed context windows limit effectiveness in multi-turn interactions requiring repeated, independent tool usage. We introduce MemTool, a short-term memory framework enabling LLM agents to dynamically manage tools or MCP server contexts across multi-turn conversations. MemTool offers three agentic architectures: 1) Autonomous Agent Mode, granting full tool management autonomy, 2) Workflow Mode, providing deterministic control without autonomy, and 3) Hybrid Mode, combining autonomous and deterministic control. Evaluating each MemTool mode across 13+ LLMs on the ScaleMCP benchmark, we conducted experiments over 100 consecutive user interactions, measuring tool removal ratios (short-term memory efficiency) and task completion accuracy. In Autonomous Agent Mode, reasoning LLMs achieve high tool-removal efficiency (90-94% over a 3-window average), while medium-sized models exhibit significantly lower efficiency (0-60%). Workflow and Hybrid modes consistently manage tool removal effectively, whereas Autonomous and Hybrid modes excel at task completion. We present trade-offs and recommendations for each MemTool mode based on task accuracy, agency, and model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22289v1">Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted for publication at SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Intent recognition is a fundamental component in task-oriented dialogue systems (TODS). Determining user intents and detecting whether an intent is Out-of-Scope (OOS) is crucial for TODS to provide reliable responses. However, traditional TODS require large amount of annotated data. In this work we propose a hybrid approach to combine BERT and LLMs in zero and few-shot settings to recognize intents and detect OOS utterances. Our approach leverages LLMs generalization power and BERT's computational efficiency in such scenarios. We evaluate our method on multi-party conversation corpora and observe that sharing information from BERT outputs to LLMs leads to system performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22286v1">Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 5 pages, 3 figures, Accepted for publication at the Second International Workshop on Construction Grammars and NLP at the 16th International Conference for Computational Semantics (IWCS) 2025
    </div>
    <details class="paper-abstract">
      The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22267v1">Promoting Online Safety by Simulating Unsafe Conversations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14907v2">GneissWeb: Preparing High Quality Data for LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models. In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0.
    </details>
</div>
