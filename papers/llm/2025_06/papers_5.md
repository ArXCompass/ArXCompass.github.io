# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13147v4">Are Your LLMs Capable of Stable Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Camera
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has shown remarkable progress in complex reasoning tasks. However, a significant disparity exists between benchmark performances and real-world applications. We attribute this gap primarily to current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, especially in complex reasoning tasks where both accuracy and consistency are essential. In this paper, we introduce G-Pass@$k$, a novel evaluation metric that continuously assesses model performance across multiple sampling attempts, quantifying both the model's performance potential and its stability. Through extensive experiments on various public and newly constructed benchmarks, we employ G-Pass@$k$ in conjunction with state-of-the-art large language models to provide comprehensive insights into their potential capabilities and operational consistency. Our findings reveal a significant opportunity to enhance the realistic reasoning abilities of LLMs, underscoring the necessity for more robust evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12212v3">Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup. The code is available at https://github.com/gszfwsb/Data-Whisperer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10084v2">Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Despite the remarkable successes of large language models (LLMs), the underlying Transformer architecture has inherent limitations in handling complex reasoning tasks. Chain-of-thought (CoT) prompting has emerged as a practical workaround, but most CoT-based methods rely on a single, generic prompt such as "think step by step", with no task-specific adaptation. These approaches expect the model to discover an effective reasoning path on its own, forcing it to search through a vast prompt space. In contrast, several studies have explored task-specific prompt designs to boost performance. However, these designs are typically developed through trial and error, lacking theoretical grounding. As a result, prompt engineering remains largely ad hoc and unguided. In this paper, we provide a theoretical framework that explains why some prompts succeed while others fail. We show that prompts function as selectors, extracting task-relevant information from the model's full hidden state during CoT reasoning. Each prompt defines a unique trajectory through the answer space, and the choice of trajectory is crucial for task performance and future navigation within the space. We analyze the complexity of finding optimal prompts and characterize the size of the prompt space for a given task. Our theory reveals principles behind effective prompt design and shows that naive CoT-using self-guided prompts like "think step by step"-can severely hinder performance. Through experiments, we show that optimal prompt search can lead to more than a 50% improvement on reasoning tasks, providing a theoretical foundation for prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11368v2">LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus and code for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03219v2">Content Moderation by LLM: From Accuracy to Legitimacy</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      One trending application of LLM (large language model) is to use it for content moderation in online platforms. Most current studies on this application have focused on the metric of accuracy -- the extent to which LLMs make correct decisions about content. This article argues that accuracy is insufficient and misleading because it fails to grasp the distinction between easy cases and hard cases, as well as the inevitable trade-offs in achieving higher accuracy. Closer examination reveals that content moderation is a constitutive part of platform governance, the key of which is to gain and enhance legitimacy. Instead of making moderation decisions correct, the chief goal of LLMs is to make them legitimate. In this regard, this article proposes a paradigm shift from the single benchmark of accuracy towards a legitimacy-based framework for evaluating the performance of LLM moderators. The framework suggests that for easy cases, the key is to ensure accuracy, speed, and transparency, while for hard cases, what matters is reasoned justification and user participation. Examined under this framework, LLMs' real potential in moderation is not accuracy improvement. Rather, LLMs can better contribute in four other aspects: to conduct screening of hard cases from easy cases, to provide quality explanations for moderation decisions, to assist human reviewers in getting more contextual information, and to facilitate user participation in a more interactive way. To realize these contributions, this article proposes a workflow for incorporating LLMs into the content moderation system. Using normative theories from law and social sciences to critically assess the new technological application, this article seeks to redefine LLMs' role in content moderation and redirect relevant research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23019v2">Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Preprint; 27 pages, 12 figures, 10 tables; Project Page at https://zeying-gong.github.io/projects/ascent
    </div>
    <details class="paper-abstract">
      Object-Goal Navigation (OGN) remains challenging in real-world, multi-floor environments and under open-vocabulary object descriptions. We observe that most episodes in widely used benchmarks such as HM3D and MP3D involve multi-floor buildings, with many requiring explicit floor transitions. However, existing methods are often limited to single-floor settings or predefined object categories. To address these limitations, we tackle two key challenges: (1) efficient cross-level planning and (2) zero-shot object-goal navigation (ZS-OGN), where agents must interpret novel object descriptions without prior exposure. We propose ASCENT, a framework that combines a Multi-Floor Spatial Abstraction module for hierarchical semantic mapping and a Coarse-to-Fine Frontier Reasoning module leveraging Large Language Models (LLMs) for context-aware exploration, without requiring additional training on new object semantics or locomotion data. Our method outperforms state-of-the-art ZS-OGN approaches on HM3D and MP3D benchmarks while enabling efficient multi-floor navigation. We further validate its practicality through real-world deployment on a quadruped robot, achieving successful object exploration across unseen floors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18792v2">REALM: A Dataset of Real-World LLM Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as the GPT series, have driven significant industrial applications, leading to economic and societal transformations. However, a comprehensive understanding of their real-world applications remains limited. To address this, we introduce REALM, a dataset of over 94,000 LLM use cases collected from Reddit and news articles. REALM captures two key dimensions: the diverse applications of LLMs and the demographics of their users. It categorizes LLM applications and explores how users' occupations relate to the types of applications they use. By integrating real-world data, REALM offers insights into LLM adoption across different domains, providing a foundation for future research on their evolving societal roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14127v2">Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing, where LLMs construct and explain answers, better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA, robustness, biases, and unfaithful explanations, showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01195v1">CoBRA: Quantifying Strategic Language Use and LLM Pragmatics</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Language is often used strategically, particularly in high-stakes, adversarial settings, yet most work on pragmatics and LLMs centers on cooperativity. This leaves a gap in systematic understanding of non-cooperative discourse. To address this, we introduce CoBRA (Cooperation-Breach Response Assessment), along with three interpretable metrics -- Benefit at Turn (BaT), Penalty at Turn (PaT), and Normalized Relative Benefit at Turn (NRBaT) -- to quantify the perceived strategic effects of discourse moves. We also present CHARM, an annotated dataset of real courtroom cross-examinations, to demonstrate the framework's effectiveness. Using these tools, we evaluate a range of LLMs and show that LLMs generally exhibit limited pragmatic understanding of strategic language. While model size shows an increase in performance on our metrics, reasoning ability does not help and largely hurts, introducing overcomplication and internal confusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01190v1">Culturally-Grounded Chain-of-Thought (CG-CoT):Enhancing LLM Performance on Culturally-Specific Tasks in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) struggle with culturally-specific reasoning tasks, particularly in low-resource languages, hindering their global applicability. Addressing this gap is crucial for equitable AI deployment. We introduce Culturally-Grounded Chain-of-Thought (CG-CoT), a novel prompting strategy that combines dense vector retrieval of cultural context with explicit reasoning sequences. Our extensive experiments on Yoruba proverb interpretation demonstrate that CG-CoT provides significantly higher culturally-aligned accuracy and depth than traditional prompting methods, validated through both automated metrics and LLM-based evaluations. Notably, we uncover stark disparities between token-level translation metrics like BLEU and human-judged cultural relevance, suggesting a rethinking of evaluation approaches for low-resource NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01140v1">ReTern: Exploiting Natural Redundancy and Sign Transformations for Enhanced Fault Tolerance in Compute-in-Memory based Ternary LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Ternary large language models (LLMs), which utilize ternary precision weights and 8-bit activations, have demonstrated competitive performance while significantly reducing the high computational and memory requirements of full-precision LLMs. The energy efficiency and performance of Ternary LLMs can be further improved by deploying them on ternary computing-in-memory (TCiM) accelerators, thereby alleviating the von-Neumann bottleneck. However, TCiM accelerators are prone to memory stuck-at faults (SAFs) leading to degradation in the model accuracy. This is particularly severe for LLMs due to their low weight sparsity. To boost the SAF tolerance of TCiM accelerators, we propose ReTern that is based on (i) fault-aware sign transformations (FAST) and (ii) TCiM bit-cell reprogramming exploiting their natural redundancy. The key idea is to utilize FAST to minimize computations errors due to SAFs in +1/-1 weights, while the natural bit-cell redundancy is exploited to target SAFs in 0 weights (zero-fix). Our experiments on BitNet b1.58 700M and 3B ternary LLMs show that our technique furnishes significant fault tolerance, notably 35% reduction in perplexity on the Wikitext dataset in the presence of faults. These benefits come at the cost of < 3%, < 7%, and < 1% energy, latency and area overheads respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01116v1">ChemAU: Harness the Reasoning of LLMs in Chemical Research with Adaptive Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used across various scenarios due to their exceptional reasoning capabilities and natural language understanding. While LLMs demonstrate strong performance in tasks involving mathematics and coding, their effectiveness diminishes significantly when applied to chemistry-related problems. Chemistry problems typically involve long and complex reasoning steps, which contain specific terminology, including specialized symbol systems and complex nomenclature conventions. These characteristics often cause general LLMs to experience hallucinations during the reasoning process due to their lack of specific knowledge. However, existing methods are struggling to effectively leverage chemical expertise and formulas. Moreover, current uncertainty estimation methods, designed to mitigate potential reasoning errors, are unable to precisely identify specific steps or key knowledge. In this work, we propose a novel framework called ChemAU, which incorporates our adaptive uncertainty estimation method that applies different uncertainty values based on the position of reasoning steps within the whole reasoning chain. Leveraging this method, ChemAU identifies gaps in chemistry knowledge and precisely supplements chemical expertise with the specialized domain model, thereby correcting and updating the previously flawed reasoning chain. Our experiments with three popular LLMs across three chemistry datasets demonstrate that ChemAU significantly enhances both reasoning accuracy and uncertainty estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01114v1">Reconsidering LLM Uncertainty Estimation Methods in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Uncertainty Estimation (UE) methods have become a crucial tool for detecting hallucinations in recent years. While numerous UE methods have been proposed, most existing studies evaluate them in isolated short-form QA settings using threshold-independent metrics such as AUROC or PRR. However, real-world deployment of UE methods introduces several challenges. In this work, we systematically examine four key aspects of deploying UE methods in practical settings. Specifically, we assess (1) the sensitivity of UE methods to decision threshold selection, (2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, (3) their applicability to long-form generation, and (4) strategies for handling multiple UE scores for a single query. Our evaluations on 19 UE methods reveal that most of them are highly sensitive to threshold selection when there is a distribution shift in the calibration dataset. While these methods generally exhibit robustness against previous chat history and typos, they are significantly vulnerable to adversarial prompts. Additionally, while existing UE methods can be adapted for long-form generation through various strategies, there remains considerable room for improvement. Lastly, ensembling multiple UE scores at test time provides a notable performance boost, which highlights its potential as a practical improvement strategy. Code is available at: https://github.com/duygunuryldz/uncertainty_in_the_wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01104v1">Contextual Candor: Enhancing LLM Trustworthiness Through Hierarchical Unanswerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The pervasive deployment of large language models (LLMs) in conversational AI systems has revolutionized information access, yet their propensity for generating factually unsupported or hallucinated responses remains a critical impediment to trustworthiness and widespread adoption. This paper introduces Reinforced Unanswerability Learning (RUL), a novel hybrid training paradigm designed to imbue LLMs with the intrinsic capability to accurately detect unanswerable questions and generate reliably appropriate responses. Unlike conventional approaches that rely on external classifiers or simple prompting, RUL integrates a discriminative unanswerability prediction head with the LLM's generative core, guided by a multi-stage learning strategy. This includes supervised fine-tuning on a novel, richly annotated dataset, Enhanced-CAsT-Answerability (ECA), which features hierarchical answerability labels and ground-truth refusal responses. Crucially, RUL incorporates a subsequent reinforcement learning with human feedback (RLHF) phase to refine the nuance, helpfulness, and informativeness of refusal responses. Extensive experiments demonstrate RUL's superior performance, achieving significantly higher accuracy in unanswerability detection across sentence, paragraph, and ranking levels, and substantially increasing the generation of appropriate refusals for unanswerable queries, alongside strong performance on answerable questions. Human evaluations further corroborate RUL's effectiveness, highlighting a marked improvement in perceived helpfulness and trustworthiness, ultimately paving the way for more reliable and user-centric conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01089v1">Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performances in tasks related to coreference resolution. However, previous studies mostly assessed LLM performance on coreference resolution with nouns and third person pronouns. This study evaluates LLM performance on coreference resolution with indexical like I, you, here and tomorrow, which come with unique challenges due to their linguistic properties. We present the first study examining how LLMs interpret indexicals in English, releasing the English Indexical Dataset with 1600 multiple-choice questions. We evaluate pioneering LLMs, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek V3. Our results reveal that LLMs exhibit an impressive performance with some indexicals (I), while struggling with others (you, here, tomorrow), and that syntactic cues (e.g. quotation) contribute to LLM performance with some indexicals, while they reduce performance with others. Code and data are available at: https://github.com/metehanoguzz/LLMs-Indexicals-English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01063v1">AI4Contracts: LLM & RAG-Powered Encoding of Financial Derivative Contracts</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 8 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) are reshaping how AI systems extract and organize information from unstructured text. A key challenge is designing AI methods that can incrementally extract, structure, and validate information while preserving hierarchical and contextual relationships. We introduce CDMizer, a template-driven, LLM, and RAG-based framework for structured text transformation. By leveraging depth-based retrieval and hierarchical generation, CDMizer ensures a controlled, modular process that aligns generated outputs with predefined schema. Its template-driven approach guarantees syntactic correctness, schema adherence, and improved scalability, addressing key limitations of direct generation methods. Additionally, we propose an LLM-powered evaluation framework to assess the completeness and accuracy of structured representations. Demonstrated in the transformation of Over-the-Counter (OTC) financial derivative contracts into the Common Domain Model (CDM), CDMizer establishes a scalable foundation for AI-driven document understanding, structured synthesis, and automated validation in broader contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v1">MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Function-calling has enabled large language models (LLMs) to act as tool-using agents, but injecting thousands of tool schemas into the prompt is costly and error-prone. We introduce MCP-Zero, a proactive agent framework that lets the LLM itself decide when and which external tools to retrieve, thereby assembling a task-specific toolchain from scratch. The framework is built upon three components: (1) Proactive Tool Request, where the model emits a structured $\left<\operatorname{tool\_assistant}\right>$ block that explicitly specifies the desired server and task; (2) Hierarchical Vector Routing, a coarse-to-fine retrieval algorithm that first selects candidate servers and then ranks tools within each server based on the semantic similarity; (3) Iterative Proactive Invocation, enabling multi-round, cross-domain toolchain construction with minimal context overhead, and allowing the model to iteratively revise its request when the returned tools are insufficient. To evaluate our approach we also compile MCP-tools, a retrieval dataset comprising 308 MCP servers and 2,797 tools extracted from the official Model-Context-Protocol repository and normalized into a unified JSON schema. Experiments show that MCP-Zero (i) effectively addresses the context overhead problem of existing methods and accurately selects the correct tool from a pool of nearly 3,000 candidates (248.1k tokens); (ii) reduces token consumption by 98\% on the APIbank while maintaining high accuracy; and (iii) supports multi-turn tool invocation with consistent accuracy across rounds. The code and dataset will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01055v1">Simple Prompt Injection Attacks Can Leak Personal Data Observed by LLM Agents During Task Execution</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 25 pages, 18 figures, NeurIPS formatting style
    </div>
    <details class="paper-abstract">
      Previous benchmarks on prompt injection in large language models (LLMs) have primarily focused on generic tasks and attacks, offering limited insights into more complex threats like data exfiltration. This paper examines how prompt injection can cause tool-calling agents to leak personal data observed during task execution. Using a fictitious banking agent, we develop data flow-based attacks and integrate them into AgentDojo, a recent benchmark for agentic security. To enhance its scope, we also create a richer synthetic dataset of human-AI banking conversations. In 16 user tasks from AgentDojo, LLMs show a 15-50 percentage point drop in utility under attack, with average attack success rates (ASR) around 20 percent; some defenses reduce ASR to zero. Most LLMs, even when successfully tricked by the attack, avoid leaking highly sensitive data like passwords, likely due to safety alignments, but they remain vulnerable to disclosing other personal data. The likelihood of password leakage increases when a password is requested along with one or two additional personal details. In an extended evaluation across 48 tasks, the average ASR is around 15 percent, with no built-in AgentDojo defense fully preventing leakage. Tasks involving data extraction or authorization workflows, which closely resemble the structure of exfiltration attacks, exhibit the highest ASRs, highlighting the interaction between task type, agent performance, and defense efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02058v1">Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Accurate evaluation of large language models (LLMs) is crucial for understanding their capabilities and guiding their development. However, current evaluations often inconsistently reflect the actual capacities of these models. In this paper, we demonstrate that one of many contributing factors to this \textit{evaluation crisis} is the oversight of unseen knowledge -- information encoded by LLMs but not directly observed or not yet observed during evaluations. We introduce KnowSum, a statistical framework designed to provide a more comprehensive assessment by quantifying the unseen knowledge for a class of evaluation tasks. KnowSum estimates the unobserved portion by extrapolating from the appearance frequencies of observed knowledge instances. We demonstrate the effectiveness and utility of KnowSum across three critical applications: estimating total knowledge, evaluating information retrieval effectiveness, and measuring output diversity. Our experiments reveal that a substantial volume of knowledge is omitted when relying solely on observed LLM performance. Importantly, KnowSum yields significantly different comparative rankings for several common LLMs based on their internal knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01049v1">Taming LLMs by Scaling Learning Rates with Gradient Grouping</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Preprint version of "Taming LLMs with Gradient Grouping" (ACL'2025). The code will be available at https://github.com/ScalingOpt/SGG
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) poses challenges due to their massive scale and heterogeneous architectures. While adaptive optimizers like AdamW help address gradient variations, they still struggle with efficient and effective parameter-wise learning rate estimation, resulting in training instability, slow convergence, and poor compatibility with parameter-efficient fine-tuning (PEFT) techniques. This work introduces Scaling with Gradient Grouping (SGG), an optimizer wrapper that improves adaptive learning rate estimation by dynamic grouping and group-specific scaling. SGG first groups gradient statistics in each layer into clusters and then applies cluster-specific scaling to calibrate learning rates for each parameter, thus imposing collective group-wise constraints while maintaining precise per-parameter adaptation. Experiments on diverse (M)LLM benchmarks show that SGG integrates seamlessly with existing optimizers, and offers consistent gains and faster convergence over baselines, with various model sizes. Its stability across varying batch sizes and learning rates establishes SGG as a robust choice for LLM optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01048v1">IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional performance across a wide range of natural language tasks. However, selecting the optimal LLM to respond to a user query often necessitates a delicate balance between performance and cost. While powerful models deliver better results, they come at a high cost, whereas smaller models are more cost-effective but less capable. To address this trade-off, we propose IRT-Router, a multi-LLM routing framework that efficiently routes user queries to the most suitable LLM. Inspired by Item Response Theory (IRT), a psychological measurement methodology, IRT-Router explicitly models the relationship between LLM capabilities and user query attributes. This not only enables accurate prediction of response performance but also provides interpretable insights, such as LLM abilities and query difficulty. Additionally, we design an online query warm-up technique based on semantic similarity, further enhancing the online generalization capability of IRT-Router. Extensive experiments on 20 LLMs and 12 datasets demonstrate that IRT-Router outperforms most baseline methods in terms of effectiveness and interpretability. Its superior performance in cold-start scenarios further confirms the reliability and practicality of IRT-Router in real-world applications. Code is available at https://github.com/Mercidaiha/IRT-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00998v1">LoRA-BAM: Input Filtering for Fine-tuned LLMs via Boxed Abstraction Monitors over LoRA Layers</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) improves performance on domain-specific tasks but can lead to overfitting, making them unreliable on out-of-distribution (OoD) queries. We propose LoRA-BAM - a method that adds OoD detection monitors to the LoRA layer using boxed abstraction to filter questions beyond the model's competence. Feature vectors from the fine-tuning data are extracted via the LLM and clustered. Clusters are enclosed in boxes; a question is flagged as OoD if its feature vector falls outside all boxes. To improve interpretability and robustness, we introduce a regularization loss during fine-tuning that encourages paraphrased questions to stay close in the feature space, and the enlargement of the decision boundary is based on the feature variance within a cluster. Our method complements existing defenses by providing lightweight and interpretable OoD detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00985v1">Do LLMs Understand Why We Write Diaries? A Method for Purpose Extraction and Clustering</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted for CompLing-2025 conference
    </div>
    <details class="paper-abstract">
      Diary analysis presents challenges, particularly in extracting meaningful information from large corpora, where traditional methods often fail to deliver satisfactory results. This study introduces a novel method based on Large Language Models (LLMs) to identify and cluster the various purposes of diary writing. By "purposes," we refer to the intentions behind diary writing, such as documenting life events, self-reflection, or practicing language skills. Our approach is applied to Soviet-era diaries (1922-1929) from the Prozhito digital archive, a rich collection of personal narratives. We evaluate different proprietary and open-source LLMs, finding that GPT-4o and o1-mini achieve the best performance, while a template-based baseline is significantly less effective. Additionally, we analyze the retrieved purposes based on gender, age of the authors, and the year of writing. Furthermore, we examine the types of errors made by the models, providing a deeper understanding of their limitations and potential areas for improvement in future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00888v1">An Integrated Platform for LEED Certification Automation Using Computer Vision and LLM-RAG</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The Leadership in Energy and Environmental Design (LEED) certification process is characterized by labor-intensive requirements for data handling, simulation, and documentation. This paper presents an automated platform designed to streamline key aspects of LEED certification. The platform integrates a PySide6-based user interface, a review Manager for process orchestration, and multiple analysis engines for credit compliance, energy modeling via EnergyPlus, and location-based evaluation. Key components include an OpenCV-based preprocessing pipeline for document analysis and a report generation module powered by the Gemma3 large language model with a retrieval-augmented generation framework. Implementation techniques - including computer vision for document analysis, structured LLM prompt design, and RAG-based report generation - are detailed. Initial results from pilot project deployment show improvements in efficiency and accuracy compared to traditional manual workflows, achieving 82% automation coverage and up to 70% reduction in documentation time. The platform demonstrates practical scalability for green building certification automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00845v1">Generalizable LLM Learning of Graph Synthetic Data with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 9 pages, 3 figures, 3 tables. Experimental code and results are publicly available at https://anonymous.4open.science/r/Graph_RL-BF08/readme.md
    </div>
    <details class="paper-abstract">
      Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph synthetic data with reinforcement learning. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that RL would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting. We employ RL algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our RL recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9\% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards, mixing synthetic and real-world task data yields potential gains, while compositionality and explainable intermediate steps remains a critical challenge even after RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00844v1">LLM Cannot Discover Causality, and Should Be Restricted to Non-Decisional Support in Causal Discovery</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      This paper critically re-evaluates LLMs' role in causal discovery and argues against their direct involvement in determining causal relationships. We demonstrate that LLMs' autoregressive, correlation-driven modeling inherently lacks the theoretical grounding for causal reasoning and introduces unreliability when used as priors in causal discovery algorithms. Through empirical studies, we expose the limitations of existing LLM-based methods and reveal that deliberate prompt engineering (e.g., injecting ground-truth knowledge) could overstate their performance, helping to explain the consistently favorable results reported in much of the current literature. Based on these findings, we strictly confined LLMs' role to a non-decisional auxiliary capacity: LLMs should not participate in determining the existence or directionality of causal relationships, but can assist the search process for causal graphs (e.g., LLM-based heuristic search). Experiments across various settings confirm that, by strictly isolating LLMs from causal decision-making, LLM-guided heuristic search can accelerate the convergence and outperform both traditional and LLM-based methods in causal structure learning. We conclude with a call for the community to shift focus from naively applying LLMs to developing specialized models and training method that respect the core principles of causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00823v1">Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 19 pages, 16 figures; accepted to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on extensive datasets that encapsulate substantial world knowledge. However, their outputs often include confidently stated inaccuracies. Earlier works suggest that LLMs encode truthfulness as a distinct linear feature, termed the "truth direction", which can classify truthfulness reliably. We address several open questions about the truth direction: (i) whether LLMs universally exhibit consistent truth directions; (ii) whether sophisticated probing techniques are necessary to identify truth directions; and (iii) how the truth direction generalizes across diverse contexts. Our findings reveal that not all LLMs exhibit consistent truth directions, with stronger representations observed in more capable models, particularly in the context of logical negation. Additionally, we demonstrate that truthfulness probes trained on declarative atomic statements can generalize effectively to logical transformations, question-answering tasks, in-context learning, and external knowledge sources. Finally, we explore the practical application of truthfulness probes in selective question-answering, illustrating their potential to improve user trust in LLM outputs. These results advance our understanding of truth directions and provide new insights into the internal representations of LLM beliefs. Our code is public at https://github.com/colored-dye/truthfulness_probe_generalization
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00807v1">Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of large language models (LLMs) have significantly advanced their performance by enabling in-depth understanding of diverse tasks. With growing interest in applying LLMs to the time series domain, this has proven nontrivial, as evidenced by the limited efficacy of straightforwardly adapting text-domain reasoning techniques. Although recent work has shown promise in several time series tasks, further leveraging advancements in LLM reasoning remains under-explored for time series classification (TSC) tasks, despite their prevalence and significance in many real-world applications. In this paper, we propose ReasonTSC, a novel framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC. Rather than straightforwardly applying existing reasoning techniques or relying solely on LLMs' built-in reasoning capabilities, ReasonTSC first steers the model to think over the essential characteristics of time series data. Next, it integrates predictions and confidence scores from plug-in classifiers, e.g., domain-specific time series models, as in-context examples. Finally, ReasonTSC guides the LLM through a structured reasoning process: it evaluates the initial assessment, backtracks to consider alternative hypotheses, and compares their merits before arriving at a final classification. Extensive experiments and systematic ablation studies demonstrate that ReasonTSC consistently outperforms both existing time series reasoning baselines and plug-in models, and is even capable of identifying and correcting plug-in models' false predictions.
    </details>
</div>
