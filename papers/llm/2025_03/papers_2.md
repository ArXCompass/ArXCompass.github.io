# llm - 2025_03

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08622v2">Policy Prototyping for LLMs: Pluralistic Alignment via Interactive and Collaborative Policymaking</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Bidirectional Human-AI Alignment (Bi-Align) Workshop @ ICLR 2025
    </div>
    <details class="paper-abstract">
      Emerging efforts in AI alignment seek to broaden participation in shaping model behavior by eliciting and integrating collective input into a policy for model finetuning. While pluralistic, these processes are often linear and do not allow participating stakeholders to confirm whether potential outcomes of their contributions are indeed consistent with their intentions. Design prototyping has long advocated for rapid iteration using tight feedback loops of ideation, experimentation, and evaluation to mitigate these issues. We thus propose policy prototyping for LLMs, a new process that draws inspiration from prototyping practices to enable stakeholders to collaboratively and interactively draft LLM policies. Through learnings from a real-world LLM policymaking initiative at an industrial AI lab, we motivate our approach and characterize policy prototyping with four guiding principles. Because policy prototyping emphasizes a contrasting set of priorities compared to previous approaches, we envision our approach to be a valuable addition to the methodological repertoire for collaborative, pluralistic alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12757v1">MAP: Multi-user Personalization with Collaborative LLM-powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA '25), April 26-May 1, 2025, Yokohama, Japan
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) and LLM-powered agents in multi-user settings underscores the need for reliable, usable methods to accommodate diverse preferences and resolve conflicting directives. Drawing on conflict resolution theory, we introduce a user-centered workflow for multi-user personalization comprising three stages: Reflection, Analysis, and Feedback. We then present MAP -- a \textbf{M}ulti-\textbf{A}gent system for multi-user \textbf{P}ersonalization -- to operationalize this workflow. By delegating subtasks to specialized agents, MAP (1) retrieves and reflects on relevant user information, while enhancing reliability through agent-to-agent interactions, (2) provides detailed analysis for improved transparency and usability, and (3) integrates user feedback to iteratively refine results. Our user study findings (n=12) highlight MAP's effectiveness and usability for conflict resolution while emphasizing the importance of user involvement in resolution verification and failure management. This work highlights the potential of multi-agent systems to implement user-centered, multi-user personalization workflows and concludes by offering insights for personalization in multi-user contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13773v1">Mitigating KV Cache Competition to Enhance User Experience in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      In Large Language Model (LLM) serving, the KV-cache (KVC) bottleneck causes high tail Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT), impairing user experience, particularly in time-sensitive applications. However, satisfying both TTFT and TBT service-level objectives (SLOs) is challenging. To address this, we propose a system, named CacheOPT for mitigating KV Cache competition, based on key insights from our measurements, incorporating novel components. First, it estimates a request's output length, bounding the deviation with a high specified probability, adjusted based on the request arrival rate. Second, it allocates the estimated KVC demand to a request, and reuses other requests' allocated KVC to avoid preemptions while reducing waiting time. Third, it proactively allocates KVC before instead of at the time a request exhausts its allocation and reserves KVC globally to prevent preemptions. Fourth, it chooses a request that has long TBT SLO, long job remaining time and short preemption time to preempt. Fifth, it selects the shortest-latency strategy between swapping and recomputation for preemptions. Experiments show that CacheOPT achieves up to 3.29$\times$ and 2.83$\times$ lower tail TBT and tail TTFT, 47\% and 53\% higher TTFT and TBT SLO attainments, and supports up to 1.58$\times$ higher request arrival rate than the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13737v1">AccelGen: Heterogeneous SLO-Guaranteed High-Throughput LLM Inference Serving for Diverse Applications</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      In this paper, we consider a mixed-prompt scenario for a large language model (LLM) inference serving system that supports diverse applications with both short prompts and long prompts and heterogeneous SLOs for iteration time. To improve throughput when handling long prompts, previous research introduces a chunking method, but has not addressed heterogeneous SLOs. To address the limitation, we propose AccelGen, a high-throughput LLM inference serving system with heterogeneous SLO guarantees for diverse applications. AccelGen introduces four core components: (1) SLO-guaranteed dynamic chunking, which dynamically adjusts chunk sizes to maximize GPU compute utilization while meeting iteration-level SLOs; (2) Iteration-level SLO-based task prioritization, which prioritizes tight-SLO requests and batches requests with similar SLOs; (3) Multi-resource-aware batching, which selects queued requests to maximize the utilizations of both GPU compute resource and key-value cache (KVC). Trace-driven real experiments demonstrate that AccelGen achieves 1.42-11.21X higher throughput, 1.43-13.71X higher goodput, 37-90% higher SLO attainment, and 1.61-12.22X lower response latency compared to the state-of-the-art approaches. It achieves performance near the Oracle, which optimally maximizes goodput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08437v2">AutoEval: Autonomous Evaluation of LLMs for Truth Maintenance and Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      This paper presents AutoEval, a novel benchmark for scaling Large Language Model (LLM) assessment in formal tasks with clear notions of correctness, such as truth maintenance in translation and logical reasoning. AutoEval is the first benchmarking paradigm that offers several key advantages necessary for scaling objective evaluation of LLMs without human labeling: (a) ability to evaluate LLMs of increasing sophistication by auto-generating tasks at different levels of difficulty; (b) auto-generation of ground truth that eliminates dependence on expensive and time-consuming human annotation; (c) the use of automatically generated, randomized datasets that mitigate the ability of successive LLMs to overfit to static datasets used in many contemporary benchmarks. Empirical analysis shows that an LLM's performance on AutoEval is highly indicative of its performance on a diverse array of other benchmarks focusing on translation and reasoning tasks, making it a valuable autonomous evaluation paradigm in settings where hand-curated datasets can be hard to obtain and/or update.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13661v1">Pensez: Less Data, Better Reasoning -- Rethinking French LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks. However, achieving strong performance in specialized domains like mathematical reasoning and non-English languages often requires extensive training on massive datasets. This paper investigates a contrasting approach: strategic fine-tuning on a small, high-quality, bilingual (English-French) dataset to enhance both the reasoning capabilities and French language proficiency of a large language model. Rather than relying on scale, we explore the hypothesis that targeted data curation and optimized training can achieve competitive, or even superior, performance. We demonstrate, through targeted supervised fine-tuning (SFT) on only 2,000 carefully selected samples, significant improvements in mathematical reasoning. Specifically, Pensez 7B exhibits an increase in accuracy of the base model up to 20% on the AIME25 and a 12% increase on a French MATH level 5 benchmark. These results challenge the prevailing assumption that massive datasets are aprerequisite for strong reasoning performance in LLMs, highlighting the potential of strategic data curation and optimized fine-tuning for enhancing both specialized skills and multilingual capabilities. Our findings have implications for the efficient development of high-performing, multilingual LLMs, especially in resource-constrained scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13657v1">Why Do Multi-Agent LLM Systems Fail?</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Despite growing enthusiasm for Multi-Agent Systems (MAS), where multiple LLM agents collaborate to accomplish tasks, their performance gains across popular benchmarks remain minimal compared to single-agent frameworks. This gap highlights the need to analyze the challenges hindering MAS effectiveness. In this paper, we present the first comprehensive study of MAS challenges. We analyze five popular MAS frameworks across over 150 tasks, involving six expert human annotators. We identify 14 unique failure modes and propose a comprehensive taxonomy applicable to various MAS frameworks. This taxonomy emerges iteratively from agreements among three expert annotators per study, achieving a Cohen's Kappa score of 0.88. These fine-grained failure modes are organized into 3 categories, (i) specification and system design failures, (ii) inter-agent misalignment, and (iii) task verification and termination. To support scalable evaluation, we integrate MASFT with LLM-as-a-Judge. We also explore if identified failures could be easily prevented by proposing two interventions: improved specification of agent roles and enhanced orchestration strategies. Our findings reveal that identified failures require more complex solutions, highlighting a clear roadmap for future research. We open-source our dataset and LLM annotator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13580v1">LLM Test Generation via Iterative Hybrid Program Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Automating unit test generation remains a significant challenge, particularly for complex methods in real-world projects. While Large Language Models (LLMs) have made strides in code generation, they struggle to achieve high branch coverage due to their limited ability to reason about intricate control flow structures. To address this limitation, we introduce Panta, a technique that emulates the iterative process human developers follow when analyzing code and constructing test cases. Panta integrates static control flow analysis and dynamic code coverage analysis to systematically guide LLMs in identifying uncovered execution paths and generating better test cases. By incorporating an iterative feedback-driven mechanism, our technique continuously refines test generation based on static and dynamic path coverage insights, ensuring more comprehensive and effective testing. Our empirical evaluation, conducted on classes with high cyclomatic complexity from open-source projects, demonstrates that Panta achieves 26% higher line coverage and 23% higher branch coverage compared to the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13572v1">VeriContaminated: Assessing LLM-Driven Verilog Coding for Data Contamination</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation, achieving exceptional results on various established benchmarking frameworks. However, concerns about data contamination - where benchmark data inadvertently leaks into pre-training or fine-tuning datasets - raise questions about the validity of these evaluations. While this issue is known, limiting the industrial adoption of LLM-driven software engineering, hardware coding has received little to no attention regarding these risks. For the first time, we analyze state-of-the-art (SOTA) evaluation frameworks for Verilog code generation (VerilogEval and RTLLM), using established methods for contamination detection (CCD and Min-K% Prob). We cover SOTA commercial and open-source LLMs (CodeGen2.5, Minitron 4b, Mistral 7b, phi-4 mini, LLaMA-{1,2,3.1}, GPT-{2,3.5,4o}, Deepseek-Coder, and CodeQwen 1.5), in baseline and fine-tuned models (RTLCoder and Verigen). Our study confirms that data contamination is a critical concern. We explore mitigations and the resulting trade-offs for code quality vs fairness (i.e., reducing contamination toward unbiased benchmarking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12686v1">Can LLMs Formally Reason as Abstract Interpreters for Program Analysis?</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated impressive capabilities in code generation and comprehension, but their potential in being able to perform program analysis in a formal, automatic manner remains under-explored. To that end, we systematically investigate whether LLMs can reason about programs using a program analysis framework called abstract interpretation. We prompt LLMs to follow two different strategies, denoted as Compositional and Fixed Point Equation, to formally reason in the style of abstract interpretation, which has never been done before to the best of our knowledge. We validate our approach using state-of-the-art LLMs on 22 challenging benchmark programs from the Software Verification Competition (SV-COMP) 2019 dataset, widely used in program analysis. Our results show that our strategies are able to elicit abstract interpretation-based reasoning in the tested models, but LLMs are susceptible to logical errors, especially while interpreting complex program structures, as well as general hallucinations. This highlights key areas for improvement in the formal reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10601v3">When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to these jailbreaking attacks, with success rates on GPT-4o escalating from 40% under ACE to 78% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12651v1">VeriLA: A Human-Centered Evaluation Framework for Interpretable Verification of LLM Agent Failures</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      AI practitioners increasingly use large language model (LLM) agents in compound AI systems to solve complex reasoning tasks, these agent executions often fail to meet human standards, leading to errors that compromise the system's overall performance. Addressing these failures through human intervention is challenging due to the agents' opaque reasoning processes, misalignment with human expectations, the complexity of agent dependencies, and the high cost of manual inspection. This paper thus introduces a human-centered evaluation framework for Verifying LLM Agent failures (VeriLA), which systematically assesses agent failures to reduce human effort and make these agent failures interpretable to humans. The framework first defines clear expectations of each agent by curating human-designed agent criteria. Then, it develops a human-aligned agent verifier module, trained with human gold standards, to assess each agent's execution output. This approach enables granular evaluation of each agent's performance by revealing failures from a human standard, offering clear guidelines for revision, and reducing human cognitive load. Our case study results show that VeriLA is both interpretable and efficient in helping practitioners interact more effectively with the system. By upholding accountability in human-agent collaboration, VeriLA paves the way for more trustworthy and human-aligned compound AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12600v1">GraphEval: A Lightweight Graph-Based LLM Framework for Idea Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      The powerful capabilities of Large Language Models (LLMs) have led to their growing use in evaluating human-generated content, particularly in evaluating research ideas within academic settings. Existing solutions primarily rely on prompt-based LLM methods or fine-tuned lightweight language models for idea evaluation. However, these methods are often unstable and struggle to comprehend the complex semantic information embedded in the ideas, impeding their ability to perform high-quality evaluations. To address the above challenges, we propose GraphEval, a lightweight graph-based LLM framework for idea evaluation. Our insight is that a complex idea can be broken down into comprehensible viewpoint nodes using prompts from small LLMs. These viewpoint nodes can then be linked together through edges created from LLM-based relation extraction and/or BERT similarity scores. The created viewpoint-graph can be used to conveniently propagate scores across view-nodes to improve the robustness of the idea evaluations. In particular, we propose two lightweight graph-based methods for idea evaluation: (1) GraphEval-LP: a training-free label propagation algorithm that propagates evaluation scores from known view-nodes to unknown nodes; (2) GraphEval-GNN: a Graph Neural Networks (GNN) that is trained to predict the evaluation scores given the observed graph with minimal computation resources. Moreover, to overcome LLM's limitation in objectively assessing the novelty of ideas, we further propose a novelty detection model to GraphEval-GNN to enhance its capability in judging idea novelty. Experiments on two datasets show GraphEval improves F1 scores by at least 14% with low computation and API costs. Additionally, GraphEval can effectively detect plagiarized ideas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12592v1">MoECollab: Democratizing LLM Development Through Collaborative Mixture of Experts</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) development has become increasingly centralized, limiting participation to well-resourced organizations. This paper introduces MoECollab, a novel framework leveraging Mixture of Experts (MoE) architecture to enable distributed, collaborative LLM development. By decomposing monolithic models into specialized expert modules coordinated by a trainable gating network, our framework allows diverse contributors to participate regardless of computational resources. We provide a complete technical implementation with mathematical foundations for expert dynamics, gating mechanisms, and integration strategies. Experiments on multiple datasets demonstrate that our approach achieves accuracy improvements of 3-7% over baseline models while reducing computational requirements by 34%. Expert specialization yields significant domain-specific gains, with improvements from 51% to 88% F1 score in general classification and from 23% to 44% accuracy in news categorization. We formalize the routing entropy optimization problem and demonstrate how proper regularization techniques lead to 14% higher expert utilization rates. These results validate MoECollab as an effective approach for democratizing LLM development through architecturally-supported collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06621v3">Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 ICLR 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong reasoning and memorization capabilities via pretraining on massive textual corpora. However, this poses risk of privacy and copyright violations, highlighting the need for efficient machine unlearning methods that remove sensitive data without retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn by reducing the likelihood of generating unwanted content, it leads to unstable optimization and catastrophic forgetting of retrained knowledge. We find that combining GA with low-rank adaptation results in poor trade-offs between computational cost and generative performance. To address these challenges, we propose two novel techniques for robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge Loss, which suppresses unwanted tokens while maintaining fluency by boosting the probability of the next most likely token. Second, we develop a data-adaptive initialization for LoRA adapters via low-rank approximation weighted with relative Fisher information, thereby focusing updates on parameters critical for removing targeted knowledge. Experiments on the Training Data Extraction Challenge dataset using GPT-Neo models as well as on the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our approach effectively removes sensitive information while maintaining reasoning and generative capabilities with minimal impact. Our implementation can be found in https://github.com/csm9493/efficient-llm-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00430v2">Evaluating Uncertainty-based Failure Detection for Closed-Loop LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 Accepted at ICRA 2024 Workshop on Back to the Future: Robot Learning Going Probabilistic. Website: https://sites.google.com/view/konwloop/home
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have witnessed remarkable performance as zero-shot task planners for robotic manipulation tasks. However, the open-loop nature of previous works makes LLM-based planning error-prone and fragile. On the other hand, failure detection approaches for closed-loop planning are often limited by task-specific heuristics or following an unrealistic assumption that the prediction is trustworthy all the time. As a general-purpose reasoning machine, LLMs or Multimodal Large Language Models (MLLMs) are promising for detecting failures. However, However, the appropriateness of the aforementioned assumption diminishes due to the notorious hullucination problem. In this work, we attempt to mitigate these issues by introducing a framework for closed-loop LLM-based planning called KnowLoop, backed by an uncertainty-based MLLMs failure detector, which is agnostic to any used MLLMs or LLMs. Specifically, we evaluate three different ways for quantifying the uncertainty of MLLMs, namely token probability, entropy, and self-explained confidence as primary metrics based on three carefully designed representative prompting strategies. With a self-collected dataset including various manipulation tasks and an LLM-based robot system, our experiments demonstrate that token probability and entropy are more reflective compared to self-explained confidence. By setting an appropriate threshold to filter out uncertain predictions and seek human help actively, the accuracy of failure detection can be significantly enhanced. This improvement boosts the effectiveness of closed-loop planning and the overall success rate of tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v3">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 Improvements: 1. Added results from refined human evaluation using VLLM and better survey methodology 2. Added independent evaluations (e.g. EQ-Bench) 3. Added citations to recent papers that have adopted/replicated min-p 4. Revised community adoption metrics for greater verifiability, focusing on major frameworks 5. Reorganised appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. However, popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures, leading to incoherent or repetitive outputs. To address this challenge, we propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by scaling according to the top token's probability. We conduct extensive experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing, demonstrating that min-p sampling improves both the quality and diversity of generated text, particularly at high temperatures. Moreover, human evaluations reveal a clear preference for min-p sampling in terms of both text quality and diversity. Min-p sampling has been adopted by multiple open-source LLM implementations, highlighting its practical utility and potential impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12556v1">From Guessing to Asking: An Approach to Resolving the Persona Knowledge Gap in LLMs during Multi-Turn Conversations</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 12 pages, 1 Figure, Oral Presentation at NAACL 2025
    </div>
    <details class="paper-abstract">
      In multi-turn dialogues, large language models (LLM) face a critical challenge of ensuring coherence while adapting to user-specific information. This study introduces the persona knowledge gap, the discrepancy between a model's internal understanding and the knowledge required for coherent, personalized conversations. While prior research has recognized these gaps, computational methods for their identification and resolution remain underexplored. We propose Conversation Preference Elicitation and Recommendation (CPER), a novel framework that dynamically detects and resolves persona knowledge gaps using intrinsic uncertainty quantification and feedback-driven refinement. CPER consists of three key modules: a Contextual Understanding Module for preference extraction, a Dynamic Feedback Module for measuring uncertainty and refining persona alignment, and a Persona-Driven Response Generation module for adapting responses based on accumulated user context. We evaluate CPER on two real-world datasets: CCPE-M for preferential movie recommendations and ESConv for mental health support. Using A/B testing, human evaluators preferred CPER's responses 42% more often than baseline models in CCPE-M and 27% more often in ESConv. A qualitative human evaluation confirms that CPER's responses are preferred for maintaining contextual relevance and coherence, particularly in longer (12+ turn) conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12547v1">LLMSeR: Enhancing Sequential Recommendation via LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Sequential Recommender Systems (SRS) have become a cornerstone of online platforms, leveraging users' historical interaction data to forecast their next potential engagement. Despite their widespread adoption, SRS often grapple with the long-tail user dilemma, resulting in less effective recommendations for individuals with limited interaction records. The advent of Large Language Models (LLMs), with their profound capability to discern semantic relationships among items, has opened new avenues for enhancing SRS through data augmentation. Nonetheless, current methodologies encounter obstacles, including the absence of collaborative signals and the prevalence of hallucination phenomena.In this work, we present LLMSeR, an innovative framework that utilizes Large Language Models (LLMs) to generate pseudo-prior items, thereby improving the efficacy of Sequential Recommender Systems (SRS). To alleviate the challenge of insufficient collaborative signals, we introduce the Semantic Interaction Augmentor (SIA), a method that integrates both semantic and collaborative information to comprehensively augment user interaction data. Moreover, to weaken the adverse effects of hallucination in SRS, we develop the Adaptive Reliability Validation (ARV), a validation technique designed to assess the reliability of the generated pseudo items. Complementing these advancements, we also devise a Dual-Channel Training strategy, ensuring seamless integration of data augmentation into the SRS training process.Extensive experiments conducted with three widely-used SRS models demonstrate the generalizability and efficacy of LLMSeR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12511v1">LLM-Driven Multi-step Translation from C to Rust using Static Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 22 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Translating software written in legacy languages to modern languages, such as C to Rust, has significant benefits in improving memory safety while maintaining high performance. However, manual translation is cumbersome, error-prone, and produces unidiomatic code. Large language models (LLMs) have demonstrated promise in producing idiomatic translations, but offer no correctness guarantees as they lack the ability to capture all the semantics differences between the source and target languages. To resolve this issue, we propose SACTOR, an LLM-driven C-to-Rust zero-shot translation tool using a two-step translation methodology: an "unidiomatic" step to translate C into Rust while preserving semantics, and an "idiomatic" step to refine the code to follow Rust's semantic standards. SACTOR utilizes information provided by static analysis of the source C program to address challenges such as pointer semantics and dependency resolution. To validate the correctness of the translated result from each step, we use end-to-end testing via the foreign function interface to embed our translated code segment into the original code. We evaluate the translation of 200 programs from two datasets and two case studies, comparing the performance of GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, Llama 3.3 70B and DeepSeek-R1 in SACTOR. Our results demonstrate that SACTOR achieves high correctness and improved idiomaticity, with the best-performing model (DeepSeek-R1) reaching 93% and (GPT-4o, Claude 3.5, DeepSeek-R1) reaching 84% correctness (on each dataset, respectively), while producing more natural and Rust-compliant translations compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12440v1">HKCanto-Eval: A Benchmark for Evaluating Cantonese Language Understanding and Cultural Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      The ability of language models to comprehend and interact in diverse linguistic and cultural landscapes is crucial. The Cantonese language used in Hong Kong presents unique challenges for natural language processing due to its rich cultural nuances and lack of dedicated evaluation datasets. The HKCanto-Eval benchmark addresses this gap by evaluating the performance of large language models (LLMs) on Cantonese language understanding tasks, extending to English and Written Chinese for cross-lingual evaluation. HKCanto-Eval integrates cultural and linguistic nuances intrinsic to Hong Kong, providing a robust framework for assessing language models in realistic scenarios. Additionally, the benchmark includes questions designed to tap into the underlying linguistic metaknowledge of the models. Our findings indicate that while proprietary models generally outperform open-weight models, significant limitations remain in handling Cantonese-specific linguistic and cultural knowledge, highlighting the need for more targeted training data and evaluation methods. The code can be accessed at https://github.com/hon9kon9ize/hkeval2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17390v2">$T^5Score$: A Methodology for Automatically Assessing the Quality of LLM Generated Multi-Document Topic Sets</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Using LLMs for Multi-Document Topic Extraction has recently gained popularity due to their apparent high-quality outputs, expressiveness, and ease of use. However, most existing evaluation practices are not designed for LLM-generated topics and result in low inter-annotator agreement scores, hindering the reliable use of LLMs for the task. To address this, we introduce $T^5Score$, an evaluation methodology that decomposes the quality of a topic set into quantifiable aspects, measurable through easy-to-perform annotation tasks. This framing enables a convenient, manual or automatic, evaluation procedure resulting in a strong inter-annotator agreement score. To substantiate our methodology and claims, we perform extensive experimentation on multiple datasets and report the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15154v2">MCCoder: Streamlining Motion Control with LLM-Assisted Code Generation and Rigorous Verification</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant potential in code generation. However, in the factory automation sector, particularly motion control, manual programming, alongside inefficient and unsafe debugging practices, remains prevalent. This stems from the complex interplay of mechanical and electrical systems and stringent safety requirements. Moreover, most current AI-assisted motion control programming efforts focus on PLCs, with little attention given to high-level languages and function libraries. To address these challenges, we introduce MCCoder, an LLM-powered system tailored for generating motion control code, integrated with a soft-motion controller. MCCoder improves code generation through a structured workflow that combines multitask decomposition, hybrid retrieval-augmented generation (RAG), and iterative self-correction, utilizing a well-established motion library. Additionally, it integrates a 3D simulator for intuitive motion validation and logs of full motion trajectories for data verification, significantly enhancing accuracy and safety. In the absence of benchmark datasets and metrics tailored for evaluating motion control code generation, we propose MCEVAL, a dataset spanning motion tasks of varying complexity. Experiments show that MCCoder outperforms baseline models using Advanced RAG, achieving an overall performance gain of 33.09% and a 131.77% improvement on complex tasks in the MCEVAL dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v1">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 51 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in \emph{social interactions} is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present \textit{Strategic Planning, Interaction, and Negotiation} (\textbf{SPIN-Bench}), a new multi-domain evaluation designed to measure the intelligence of \emph{strategic planning} and \emph{social reasoning}. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also \emph{conceptual inference} of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle \emph{basic fact retrieval} and \emph{short-range planning} reasonably well, they encounter significant performance bottlenecks in tasks requiring \emph{deep multi-hop reasoning} over large state spaces and \emph{socially adept} coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12347v1">Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Synthetic data offers a promising path to train models while preserving data privacy. Differentially private (DP) finetuning of large language models (LLMs) as data generator is effective, but is impractical when computation resources are limited. Meanwhile, prompt-based methods such as private evolution, depend heavily on the manual prompts, and ineffectively use private information in their iterative data selection process. To overcome these limitations, we propose CTCL (Data Synthesis with ConTrollability and CLustering), a novel framework for generating privacy-preserving synthetic data without extensive prompt engineering or billion-scale LLM finetuning. CTCL pretrains a lightweight 140M conditional generator and a clustering-based topic model on large-scale public data. To further adapt to the private domain, the generator is DP finetuned on private data for fine-grained textual information, while the topic model extracts a DP histogram representing distributional information. The DP generator then samples according to the DP histogram to synthesize a desired number of data examples. Evaluation across five diverse domains demonstrates the effectiveness of our framework, particularly in the strong privacy regime. Systematic ablation validates the design of each framework component and highlights the scalability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12344v1">EXPRESS: An LLM-Generated Explainable Property Valuation System with Neighbor Imputation</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The demand for property valuation has attracted significant attention from sellers, buyers, and customers applying for loans. Reviews of existing approaches have revealed shortcomings in terms of not being able to handle missing value situations, as well as lacking interpretability, which means they cannot be used in real-world applications. To address these challenges, we propose an LLM-Generated EXplainable PRopErty valuation SyStem with neighbor imputation called EXPRESS, which provides the customizable missing value imputation technique, and addresses the opaqueness of prediction by providing the feature-wise explanation generated by LLM. The dynamic nearest neighbor search finds similar properties depending on different application scenarios by property configuration set by users (e.g., house age as criteria for the house in rural areas, and locations for buildings in urban areas). Motivated by the human appraisal procedure, we generate feature-wise explanations to provide users with a more intuitive understanding of the prediction results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12340v1">SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 NAACL 2025; Code available at https://github.com/AIoT-MLSys-Lab/SVD-LLM
    </div>
    <details class="paper-abstract">
      Despite significant advancements, the practical deployment of Large Language Models (LLMs) is often hampered by their immense sizes, highlighting the need for effective compression techniques. Singular Value Decomposition (SVD) is a promising LLM compression technique. However, existing SVD-based compression methods fall short in reducing truncation losses, leading to less competitive performance in compressed models. In this work, we introduce SVD-LLM V2, a SVD-based LLM compression method that optimizes singular value truncation in SVD compression with two techniques. First, SVD-LLM V2 proposes to use theoretical truncation loss of weight matrices to assign a unique compression ratio to each weight matrix at different layers to accommodate weight redundancy heterogeneity. Second, SVD-LLM V2 proposes loss-optimized weight truncation to ensure that the truncated singular values result in a lower and more stable truncation loss in practice. We evaluate SVD-LLM V2 on ten datasets and five LLMs at various scales. Our results show SVD-LLM V2 outperforms state-of-the-art SVD-based LLM compression methods. Our code is available at https://github.com/AIoT-MLSys-Lab/SVD-LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12334v1">When neural implant meets multimodal LLM: A dual-loop system for neuromodulation and naturalistic neuralbehavioral research</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      We propose a novel dual-loop system that synergistically combines responsive neurostimulation (RNS) implants with artificial intelligence-driven wearable devices for treating post-traumatic stress disorder (PTSD) and enabling naturalistic brain research. In PTSD Therapy Mode, an implanted closed-loop neural device monitors amygdala activity and provides on-demand stimulation upon detecting pathological theta oscillations, while an ensemble of wearables (smart glasses, smartwatches, smartphones) uses multimodal large language model (LLM) analysis of sensory data to detect environmental or physiological PTSD triggers and deliver timely audiovisual interventions. Logged events from both the neural and wearable loops are analyzed to personalize trigger detection and progressively transition patients to non-invasive interventions. In Neuroscience Research Mode, the same platform is adapted for real-world brain activity capture. Wearable-LLM systems recognize naturalistic events (social interactions, emotional situations, compulsive behaviors, decision making) and signal implanted RNS devices (via wireless triggers) to record synchronized intracranial data during these moments. This approach builds on recent advances in mobile intracranial EEG recording and closed-loop neuromodulation in humans (BRAIN Initiative, 2023) (Mobbs et al., 2021). We discuss how our interdisciplinary system could revolutionize PTSD therapy and cognitive neuroscience by enabling 24/7 monitoring, context-aware intervention, and rich data collection outside traditional labs. The vision is a future where AI-enhanced devices continuously collaborate with the human brain, offering therapeutic support and deep insights into neural function, with the resulting real-world context rich neural data, in turn, accelerating the development of more biologically-grounded and human-centric AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12329v1">CapArena: Benchmarking and Analyzing Detailed Image Captioning in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Image captioning has been a longstanding challenge in vision-language research. With the rise of LLMs, modern Vision-Language Models (VLMs) generate detailed and comprehensive image descriptions. However, benchmarking the quality of such captions remains unresolved. This paper addresses two key questions: (1) How well do current VLMs actually perform on image captioning, particularly compared to humans? We built CapArena, a platform with over 6000 pairwise caption battles and high-quality human preference votes. Our arena-style evaluation marks a milestone, showing that leading models like GPT-4o achieve or even surpass human performance, while most open-source models lag behind. (2) Can automated metrics reliably assess detailed caption quality? Using human annotations from CapArena, we evaluate traditional and recent captioning metrics, as well as VLM-as-a-Judge. Our analysis reveals that while some metrics (e.g., METEOR) show decent caption-level agreement with humans, their systematic biases lead to inconsistencies in model ranking. In contrast, VLM-as-a-Judge demonstrates robust discernment at both the caption and model levels. Building on these insights, we release CapArena-Auto, an accurate and efficient automated benchmark for detailed captioning, achieving 94.3% correlation with human rankings at just $4 per test. Data and resources will be open-sourced at https://caparena.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12326v1">Leveraging Vision Capabilities of Multimodal LLMs for Automated Data Extraction from Plots</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Automated data extraction from research texts has been steadily improving, with the emergence of large language models (LLMs) accelerating progress even further. Extracting data from plots in research papers, however, has been such a complex task that it has predominantly been confined to manual data extraction. We show that current multimodal large language models, with proper instructions and engineered workflows, are capable of accurately extracting data from plots. This capability is inherent to the pretrained models and can be achieved with a chain-of-thought sequence of zero-shot engineered prompts we call PlotExtract, without the need to fine-tune. We demonstrate PlotExtract here and assess its performance on synthetic and published plots. We consider only plots with two axes in this analysis. For plots identified as extractable, PlotExtract finds points with over 90% precision (and around 90% recall) and errors in x and y position of around 5% or lower. These results prove that multimodal LLMs are a viable path for high-throughput data extraction for plots and in many circumstances can replace the current manual methods of data extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18771v2">CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 Extended version currently under review (Workshop version: HEAL at CHI 2024)
    </div>
    <details class="paper-abstract">
      Existing LLM-as-a-Judge approaches for evaluating text generation suffer from rating inconsistencies, with low agreement and high rating variance across different evaluator models. We attribute this to subjective evaluation criteria combined with Likert scale scoring in existing protocols. To address this issue, we introduce CheckEval, a checklist-based evaluation framework that improves rating reliability via decomposed binary questions. Through experiments with 12 evaluator models across multiple datasets, we first demonstrate that CheckEval strongly correlates with human judgments, improving the average correlation with human judgments by 0.10. More importantly, CheckEval dramatically improves the average agreement across evaluator models by 0.45 and reduces the score variance. CheckEval scores furthermore have the benefit of being more interpretable because it decomposes evaluation criteria into traceable binary decisions, allowing analyses of specific attributes driving quality judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13554v1">LLMs' Leaning in European Elections</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Many studies suggest that LLMs have left wing leans. The article extends the US presidential election analysis made in previous works, where multiple LLMs were asked to vote between Joe Biden and Donald Trump in a virtual election, and the results showed a clear lean of LLMs toward Joe Biden. This article considers natural follow-up questions that could arise from that experiment, such as: what is the extent of this phenomenon? Is it generalizable to multiple virtual elections in other countries? The article considers virtual elections in ten european countries: Germany, France, Italy, Spain, Poland, Romania, Netherlands, Belgium, Czech Republic, and Sweden, and with four different LLMs: gpt4o, claude 3.5 sonnet, mistral-large, and gemini-2.0-flash.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13553v1">LLM-Mediated Guidance of MARL Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 31 pages, 50 figures
    </div>
    <details class="paper-abstract">
      In complex multi-agent environments, achieving efficient learning and desirable behaviours is a significant challenge for Multi-Agent Reinforcement Learning (MARL) systems. This work explores the potential of combining MARL with Large Language Model (LLM)-mediated interventions to guide agents toward more desirable behaviours. Specifically, we investigate how LLMs can be used to interpret and facilitate interventions that shape the learning trajectories of multiple agents. We experimented with two types of interventions, referred to as controllers: a Natural Language (NL) Controller and a Rule-Based (RB) Controller. The NL Controller, which uses an LLM to simulate human-like interventions, showed a stronger impact than the RB Controller. Our findings indicate that agents particularly benefit from early interventions, leading to more efficient training and higher performance. Both intervention types outperform the baseline without interventions, highlighting the potential of LLM-mediated guidance to accelerate training and enhance MARL performance in challenging environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04636v2">Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 Accepted by the ICLR 2025 Workshop on GenAI Watermarking
    </div>
    <details class="paper-abstract">
      As open-source large language models (LLMs) like Llama3 become more capable, it is crucial to develop watermarking techniques to detect their potential misuse. Existing watermarking methods either add watermarks during LLM inference, which is unsuitable for open-source LLMs, or primarily target classification LLMs rather than recent generative LLMs. Adapting these watermarks to open-source LLMs for misuse detection remains an open challenge. This work defines two misuse scenarios for open-source LLMs: intellectual property (IP) violation and LLM Usage Violation. Then, we explore the application of inference-time watermark distillation and backdoor watermarking in these contexts. We propose comprehensive evaluation methods to assess the impact of various real-world further fine-tuning scenarios on watermarks and the effect of these watermarks on LLM performance. Our experiments reveal that backdoor watermarking could effectively detect IP Violation, while inference-time watermark distillation is applicable in both scenarios but less robust to further fine-tuning and has a more significant impact on LLM performance compared to backdoor watermarking. Exploring more advanced watermarking methods for open-source LLMs to detect their misuse should be an important future direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06858v2">Taming Sensitive Weights : Noise Perturbation Fine-tuning for Robust LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 Accepted as poster by CPAL2025
    </div>
    <details class="paper-abstract">
      Quantization is a critical step to enable efficient LLM serving under limited resource. However, previous research observes that certain weights in the LLM, known as outliers, are significantly sensitive to quantization noises. Existing quantization methods leave these outliers as floating points or higher precisions to retain performance, posting challenges on the efficient hardware deployment of the mixed-precision model. This work investigates an alternative way to tame the sensitive weights' impact on the quantization error, by reducing the loss Hessian trace with respect to outliers through an efficient fine-tuning process. We propose Noise Perturbation Fine-tuning (NPFT), which identifies outlier weights and add random weight perturbations on the outliers as the model going through a PEFT optimization. NPFT tames the sensitivity of outlier weights so that the quantized model performance can be improved without special treatment to the outliers. When applied to OPT and LLaMA models, our NPFT method achieves stable performance improvements for both uniform and non-uniform quantizers, while also offering better inference efficiency. Notably, the simplest RTN can achieve performance on par with GPTQ using our NPFT on LLaMA2-7B-4bits benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12225v1">Interpretation Gaps in LLM-Assisted Comprehension of Privacy Documents</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      This article explores the gaps that can manifest when using a large language model (LLM) to obtain simplified interpretations of data practices from a complex privacy policy. We exemplify these gaps to showcase issues in accuracy, completeness, clarity and representation, while advocating for continued research to realize an LLM's true potential in revolutionizing privacy management through personal assistants and automated compliance checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12217v1">TFHE-Coder: Evaluating LLM-agentic Fully Homomorphic Encryption Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Fully Homomorphic Encryption over the torus (TFHE) enables computation on encrypted data without decryption, making it a cornerstone of secure and confidential computing. Despite its potential in privacy preserving machine learning, secure multi party computation, private blockchain transactions, and secure medical diagnostics, its adoption remains limited due to cryptographic complexity and usability challenges. While various TFHE libraries and compilers exist, practical code generation remains a hurdle. We propose a compiler integrated framework to evaluate LLM inference and agentic optimization for TFHE code generation, focusing on logic gates and ReLU activation. Our methodology assesses error rates, compilability, and structural similarity across open and closedsource LLMs. Results highlight significant limitations in off-the-shelf models, while agentic optimizations such as retrieval augmented generation (RAG) and few-shot prompting reduce errors and enhance code fidelity. This work establishes the first benchmark for TFHE code generation, demonstrating how LLMs, when augmented with domain-specific feedback, can bridge the expertise gap in FHE code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12185v1">FAILS: A Framework for Automated Collection and Analysis of LLM Service Incidents</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) services such as ChatGPT, DALLE, and Cursor have quickly become essential for society, businesses, and individuals, empowering applications such as chatbots, image generation, and code assistance. The complexity of LLM systems makes them prone to failures and affects their reliability and availability, yet their failure patterns are not fully understood, making it an emerging problem. However, there are limited datasets and studies in this area, particularly lacking an open-access tool for analyzing LLM service failures based on incident reports. Addressing these problems, in this work we propose FAILS, the first open-sourced framework for incident reports collection and analysis on different LLM services and providers. FAILS provides comprehensive data collection, analysis, and visualization capabilities, including:(1) It can automatically collect, clean, and update incident data through its data scraper and processing components;(2) It provides 17 types of failure analysis, allowing users to explore temporal trends of incidents, analyze service reliability metrics, such as Mean Time to Recovery (MTTR) and Mean Time Between Failures (MTBF);(3) It leverages advanced LLM tools to assist in data analysis and interpretation, enabling users to gain observations and insights efficiently. All functions are integrated in the backend, allowing users to easily access them through a web-based frontend interface. FAILS supports researchers, engineers, and general users to understand failure patterns and further mitigate operational incidents and outages in LLM services. The framework is publicly available on https://github.com/atlarge-research/FAILS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01001v3">Towards An Efficient LLM Training Paradigm for CTR Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated tremendous potential as the next-generation ranking-based recommendation system. Many recent works have shown that LLMs can significantly outperform conventional click-through-rate (CTR) prediction approaches. Despite such promising results, the computational inefficiency inherent in the current training paradigm makes it particularly challenging to train LLMs for ranking-based recommendation tasks on large datasets. To train LLMs for CTR prediction, most existing studies adopt the prevalent ''sliding-window'' paradigm. Given a sequence of $m$ user interactions, a unique training prompt is constructed for each interaction by designating it as the prediction target along with its preceding $n$ interactions serving as context. In turn, the sliding-window paradigm results in an overall complexity of $O(mn^2)$ that scales linearly with the length of user interactions. Consequently, a direct adoption to train LLMs with such strategy can result in prohibitively high training costs as the length of interactions grows. To alleviate the computational inefficiency, we propose a novel training paradigm, namely Dynamic Target Isolation (DTI), that structurally parallelizes the training of $k$ (where $k >> 1$) target interactions. Furthermore, we identify two major bottlenecks - hidden-state leakage and positional bias overfitting - that limit DTI to only scale up to a small value of $k$ (e.g., 5) then propose a computationally light solution to effectively tackle each. Through extensive experiments on three widely adopted public CTR datasets, we empirically show that DTI reduces training time by an average of $\textbf{92%}$ (e.g., from $70.5$ hrs to $5.31$ hrs), without compromising CTR prediction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12152v1">Improving LLM-based Document-level Machine Translation with Multi-Knowledge Fusion</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Recent studies in prompting large language model (LLM) for document-level machine translation (DMT) primarily focus on the inter-sentence context by flatting the source document into a long sequence. This approach relies solely on the sequence of sentences within the document. However, the complexity of document-level sequences is greater than that of shorter sentence-level sequences, which may limit LLM's ability in DMT when only this single-source knowledge is used. In this paper, we propose an enhanced approach by incorporating multiple sources of knowledge, including both the document summarization and entity translation, to enhance the performance of LLM-based DMT. Given a source document, we first obtain its summarization and translation of entities via LLM as the additional knowledge. We then utilize LLMs to generate two translations of the source document by fusing these two single knowledge sources, respectively. Finally, recognizing that different sources of knowledge may aid or hinder the translation of different sentences, we refine and rank the translations by leveraging a multi-knowledge fusion strategy to ensure the best results. Experimental results in eight document-level translation tasks show that our approach achieves an average improvement of 0.8, 0.6, and 0.4 COMET scores over the baseline without extra knowledge for LLaMA3-8B-Instruct, Mistral-Nemo-Instruct, and GPT-4o-mini, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12123v1">MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 Under review. Project page:https://sabijun.github.io/MT_RewardTreePage
    </div>
    <details class="paper-abstract">
      Process reward models (PRMs) have shown success in complex reasoning tasks for large language models (LLMs). However, their application to machine translation (MT) remains underexplored due to the lack of systematic methodologies and evaluation benchmarks. To address this gap, we introduce \textbf{MT-RewardTree}, a comprehensive framework for constructing, evaluating, and deploying process reward models in MT. Unlike traditional vanilla preference pair construction, we propose a novel method for automatically generating token-level preference pairs using approximate Monte Carlo Tree Search (MCTS), which mitigates the prohibitive cost of human annotation for fine-grained steps. Then, we establish the first MT-specific reward model benchmark and provide a systematic comparison of different reward modeling architectures, revealing that token-level supervision effectively captures fine-grained preferences. Experimental results demonstrate that our MT-PRM-Qwen-2.5-3B achieves state-of-the-art performance in both token-level and sequence-level evaluation given the same input prefix. Furthermore, we showcase practical applications where PRMs enable test-time alignment for LLMs without additional alignment training and significantly improve performance in hypothesis ensembling. Our work provides valuable insights into the role of reward models in MT research. Our code and data are released in \href{https://sabijun.github.io/MT_RewardTreePage/}{https://sabijun.github.io/MT\_RewardTreePage}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09291v2">LAVCap: LLM-based Audio-Visual Captioning using Optimal Transport</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 5 pages, 2 figures; Accepted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Automated audio captioning is a task that generates textual descriptions for audio content, and recent studies have explored using visual information to enhance captioning quality. However, current methods often fail to effectively fuse audio and visual data, missing important semantic cues from each modality. To address this, we introduce LAVCap, a large language model (LLM)-based audio-visual captioning framework that effectively integrates visual information with audio to improve audio captioning performance. LAVCap employs an optimal transport-based alignment loss to bridge the modality gap between audio and visual features, enabling more effective semantic extraction. Additionally, we propose an optimal transport attention module that enhances audio-visual fusion using an optimal transport assignment map. Combined with the optimal training strategy, experimental results demonstrate that each component of our framework is effective. LAVCap outperforms existing state-of-the-art methods on the AudioCaps dataset, without relying on large datasets or post-processing. Code is available at https://github.com/NAVER-INTEL-Co-Lab/gaudi-lavcap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17388v2">Can LLMs be Good Graph Judger for Knowledge Graph Construction?</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      In real-world scenarios, most of the data obtained from information retrieval (IR) system is unstructured. Converting natural language sentences into structured Knowledge Graphs (KGs) remains a critical challenge. The quality of constructed KGs may also impact the performance of some KG-dependent domains like GraphRAG systems and recommendation systems. Recently, Large Language Models (LLMs) have demonstrated impressive capabilities in addressing a wide range of natural language processing tasks. However, there are still challenges when utilizing LLMs to address the task of generating structured KGs. And we have identified three limitations with respect to existing KG construction methods. (1)There is a large amount of information and excessive noise in real-world documents, which could result in extracting messy information. (2)Native LLMs struggle to effectively extract accuracy knowledge from some domain-specific documents. (3)Hallucinations phenomenon cannot be overlooked when utilizing LLMs directly as an unsupervised method for constructing KGs. In this paper, we propose GraphJudger, a knowledge graph construction framework to address the aforementioned challenges. We introduce three innovative modules in our method, which are entity-centric iterative text denoising, knowledge aware instruction tuning and graph judgement, respectively. We seek to utilize the capacity of LLMs to function as a graph judger, a capability superior to their role only as a predictor for KG construction problems. Experiments conducted on two general text-graph pair datasets and one domain-specific text-graph pair dataset show superior performances compared to baseline methods. The code of our proposed method is available at https://github.com/hhy-huang/GraphJudger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04779v3">Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to automate programming tasks. Yet, LLMs' capabilities in reasoning about program semantics are still inadequately studied, leaving significant potential for further exploration. This paper introduces FormalBench, a comprehensive benchmark designed to evaluate LLMs' reasoning abilities on program semantics, particularly via the task of synthesizing formal program specifications to assist verifying program correctness. This task requires both comprehensive reasoning over all possible program executions and the generation of precise, syntactically correct expressions that adhere to formal syntax and semantics. Using this benchmark, we evaluated the ability of LLMs in synthesizing consistent and complete specifications. Our findings show that LLMs perform well with simple control flows but struggle with more complex structures, especially loops, even with advanced prompting. Additionally, LLMs exhibit limited robustness against semantic-preserving transformations. We also highlight common failure patterns and design self-repair prompts, improving success rates by 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12043v1">An LLM-Integrated Framework for Completion, Management, and Tracing of STPA</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      In many safety-critical engineering domains, hazard analysis techniques are an essential part of requirement elicitation. Of the methods proposed for this task, STPA (System-Theoretic Process Analysis) represents a relatively recent development in the field. The completion, management, and traceability of this hazard analysis technique present a time-consuming challenge to the requirements and safety engineers involved. In this paper, we introduce a free, open-source software framework to build STPA models with several automated workflows powered by large language models (LLMs). In past works, LLMs have been successfully integrated into a myriad of workflows across various fields. Here, we demonstrate that LLMs can be used to complete tasks associated with STPA with a high degree of accuracy, saving the time and effort of the human engineers involved. We experimentally validate our method on real-world STPA models built by requirement engineers and researchers. The source code of our software framework is available at the following link: https://github.com/blueskysolarracing/stpa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08227v3">DALL-M: Context-Aware Clinical Data Augmentation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      X-ray images are vital in medical diagnostics, but their effectiveness is limited without clinical context. Radiologists often find chest X-rays insufficient for diagnosing underlying diseases, necessitating the integration of structured clinical features with radiology reports. To address this, we introduce DALL-M, a novel framework that enhances clinical datasets by generating contextual synthetic data. DALL-M augments structured patient data, including vital signs (e.g., heart rate, oxygen saturation), radiology findings (e.g., lesion presence), and demographic factors. It integrates this tabular data with contextual knowledge extracted from radiology reports and domain-specific resources (e.g., Radiopaedia, Wikipedia), ensuring clinical consistency and reliability. DALL-M follows a three-phase process: (i) clinical context storage, (ii) expert query generation, and (iii) context-aware feature augmentation. Using large language models (LLMs), it generates both contextual synthetic values for existing clinical features and entirely new, clinically relevant features. Applied to 799 cases from the MIMIC-IV dataset, DALL-M expanded the original 9 clinical features to 91. Empirical validation with machine learning models (including Decision Trees, Random Forests, XGBoost, and TabNET) demonstrated a 16.5% improvement in F1 score and a 25% increase in Precision and Recall. DALL-M bridges an important gap in clinical data augmentation by preserving data integrity while enhancing predictive modeling in healthcare. Our results show that integrating LLM-generated synthetic features significantly improves model performance, making DALL-M a scalable and practical approach for AI-driven medical diagnostics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05864v3">Permute-and-Flip: An optimally stable and watermarkable decoder for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys stability properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-stability tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson (2023)'s Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and its Gumbel watermarked counterpart) in terms of perplexity, while retaining the same stability (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06327v2">Unveiling Inefficiencies in LLM-Generated Code: Toward a Comprehensive Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely adopted for automated code generation with promising results. Although prior research has assessed LLM-generated code and identified various quality issues -- such as redundancy, poor maintainability, and sub-optimal performance a systematic understanding and categorization of these inefficiencies remain unexplored. Without such knowledge, practitioners struggle to optimize LLM-generated code for real-world applications, limiting its adoption. This study can also guide improving code LLMs, enhancing the quality and efficiency of code generation. Therefore, in this study, we empirically investigate inefficiencies in LLM-generated code by state-of-the-art models, i.e., CodeLlama, DeepSeek-Coder, and CodeGemma. To do so, we analyze 492 generated code snippets in the HumanEval++ dataset. We then construct a taxonomy of inefficiencies in LLM-generated code that includes 5 categories General Logic, Performance, Readability, Maintainability, and Errors) and 19 subcategories of inefficiencies. We then validate the proposed taxonomy through an online survey with 58 LLM practitioners and researchers. Our study indicates that logic and performance-related inefficiencies are the most popular, relevant, and frequently co-occur and impact overall code quality inefficiency. Our taxonomy provides a structured basis for evaluating the quality LLM-generated code and guiding future research to improve code generation efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11985v1">No LLM is Free From Bias: A Comprehensive Study of Bias Evaluation in Large Language models</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Advancements in Large Language Models (LLMs) have increased the performance of different natural language understanding as well as generation tasks. Although LLMs have breached the state-of-the-art performance in various tasks, they often reflect different forms of bias present in the training data. In the light of this perceived limitation, we provide a unified evaluation of benchmarks using a set of representative LLMs that cover different forms of biases starting from physical characteristics to socio-economic categories. Moreover, we propose five prompting approaches to carry out the bias detection task across different aspects of bias. Further, we formulate three research questions to gain valuable insight in detecting biases in LLMs using different approaches and evaluation metrics across benchmarks. The results indicate that each of the selected LLMs suffer from one or the other form of bias with the LLaMA3.1-8B model being the least biased. Finally, we conclude the paper with the identification of key challenges and possible future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02683v3">DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 Accepted into ICLR 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      As users increasingly seek guidance from LLMs for decision-making in daily life, many of these decisions are not clear-cut and depend significantly on the personal values and ethical standards of people. We present DailyDilemmas, a dataset of 1,360 moral dilemmas encountered in everyday life. Each dilemma presents two possible actions, along with affected parties and relevant human values for each action. Based on these dilemmas, we gather a repository of human values covering diverse everyday topics, such as interpersonal relationships, workplace, and environmental issues. With DailyDilemmas, we evaluate LLMs on these dilemmas to determine what action they will choose and the values represented by these action choices. Then, we analyze values through the lens of five theoretical frameworks inspired by sociology, psychology, and philosophy, including the World Values Survey, Moral Foundations Theory, Maslow's Hierarchy of Needs, Aristotle's Virtues, and Plutchik's Wheel of Emotions. For instance, we find LLMs are most aligned with self-expression over survival in World Values Survey and care over loyalty in Moral Foundations Theory. Interestingly, we find substantial preference differences in models for some core values. For example, for truthfulness, Mixtral-8x7B neglects it by 9.7% while GPT-4-turbo selects it by 9.4%. We also study the recent guidance released by OpenAI (ModelSpec), and Anthropic (Constitutional AI) to understand how their designated principles reflect their models' actual value prioritization when facing nuanced moral reasoning in daily-life settings. Finally, we find that end users cannot effectively steer such prioritization using system prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11951v1">SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 13 pages, 8 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Recent LLM-based agent frameworks have demonstrated impressive capabilities in task delegation and workflow orchestration, but face significant challenges in maintaining context awareness and ensuring planning consistency. This paper presents SagaLLM, a structured multi-agent framework that addresses four fundamental limitations in current LLM approaches: inadequate self-validation, context narrowing, lacking transaction properties, and insufficient inter-agent coordination. By implementing specialized context management agents and validation protocols, SagaLLM preserves critical constraints and state information throughout complex planning processes, enabling robust and consistent decision-making even during disruptions. We evaluate our approach using selected problems from the REALM benchmark, focusing on sequential and reactive planning scenarios that challenge both context retention and adaptive reasoning. Our experiments with state-of-the-art LLMs, Claude 3.7, DeepSeek R1, GPT-4o, and GPT-o1, demonstrate that while these models exhibit impressive reasoning capabilities, they struggle with maintaining global constraint awareness during complex planning tasks, particularly when adapting to unexpected changes. In contrast, the distributed cognitive architecture of SagaLLM shows significant improvements in planning consistency, constraint enforcement, and adaptation to disruptions in various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10766v4">JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 40 pages, 12 figures
    </div>
    <details class="paper-abstract">
      The systems and software powered by Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous scenarios. However, current LLM systems are vulnerable to prompt-based attacks, with jailbreaking attacks enabling the LLM system to generate harmful content, while hijacking attacks manipulate the LLM system to perform attacker-desired tasks, underscoring the necessity for detection tools. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework deployed on top of LLM systems for prompt-based attacks across text and image modalities. JailGuard operates on the principle that attacks are inherently less robust than benign ones. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the target model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. The evaluation on the dataset containing 15 known attack types suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11617v1">ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 19 pages, multiple figures
    </div>
    <details class="paper-abstract">
      Analysis and comprehension of assembly code are crucial in various applications, such as reverse engineering. However, the low information density and lack of explicit syntactic structures in assembly code pose significant challenges. Pioneering approaches with masked language modeling (MLM)-based methods have been limited by facilitating natural language interaction. While recent methods based on decoder-focused large language models (LLMs) have significantly enhanced semantic representation, they still struggle to capture the nuanced and sparse semantics in assembly code. In this paper, we propose Assembly Augmented Tuning (ASMA-Tune), an end-to-end structural-semantic instruction-tuning framework. Our approach synergizes encoder architectures with decoder-based LLMs through projector modules to enable comprehensive code understanding. Experiments show that ASMA-Tune outperforms existing benchmarks, significantly enhancing assembly code comprehension and instruction-following abilities. Our model and dataset are public at https://github.com/wxy3596/ASMA-Tune.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11614v1">Neutralizing Bias in LLM Reasoning using Entailment Graphs</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      LLMs are often claimed to be capable of Natural Language Inference (NLI), which is widely regarded as a cornerstone of more complex forms of reasoning. However, recent works show that LLMs still suffer from hallucinations in NLI due to attestation bias, where LLMs overly rely on propositional memory to build shortcuts. To solve the issue, we design an unsupervised framework to construct counterfactual reasoning data and fine-tune LLMs to reduce attestation bias. To measure bias reduction, we build bias-adversarial variants of NLI datasets with randomly replaced predicates in premises while keeping hypotheses unchanged. Extensive evaluations show that our framework can significantly reduce hallucinations from attestation bias. Then, we further evaluate LLMs fine-tuned with our framework on original NLI datasets and their bias-neutralized versions, where original entities are replaced with randomly sampled ones. Extensive results show that our framework consistently improves inferential performance on both original and bias-neutralized NLI datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11586v1">Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs using Semantic Space</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 ICLR 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in chatbots or AI assistants to hold conversations with a human user. In such applications, the quality (e.g., user engagement, safety) of a conversation is important and can only be exactly known at the end of the conversation. To maximize its expected quality, conversation planning reasons about the stochastic transitions within a conversation to select the optimal LLM response at each turn. Existing simulation-based conversation planning algorithms typically select the optimal response by simulating future conversations with a large number of LLM queries at every turn. However, this process is extremely time-consuming and hence impractical for real-time conversations. This paper presents a novel approach called Semantic space COnversation Planning with improved Efficiency (SCOPE) that exploits the dense semantic representation of conversations to perform conversation planning efficiently. In particular, SCOPE models the stochastic transitions in conversation semantics and their associated rewards to plan entirely within the semantic space. This allows us to select the optimal LLM response at every conversation turn without needing additional LLM queries for simulation. As a result, SCOPE can perform conversation planning 70 times faster than conventional simulation-based planning algorithms when applied to a wide variety of conversation starters and two reward functions seen in the real world, yet achieving a higher reward within a practical planning budget. Our code can be found at: https://github.com/chenzhiliang94/convo-plan-SCOPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16560v2">Dynamic-Width Speculative Beam Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown outstanding performance across numerous real-world tasks. However, the autoregressive nature of these models makes the inference process slow and costly. Speculative decoding has emerged as a promising solution, leveraging a smaller auxiliary model to draft future tokens, which are then validated simultaneously by the larger model, achieving a speed-up of 1-2x. Although speculative decoding matches the same distribution as multinomial sampling, multinomial sampling itself is prone to suboptimal outputs, whereas beam sampling is widely recognized for producing higher-quality results by maintaining multiple candidate sequences at each step. This paper explores the novel integration of speculative decoding with beam sampling. However, there are four key challenges: (1) how to generate multiple sequences from the larger model's distribution given drafts sequences from the small model; (2) how to dynamically optimize the number of beams to balance efficiency and accuracy; (3) how to efficiently verify the multiple drafts in parallel; and (4) how to address the extra memory costs inherent in beam sampling. To address these challenges, we propose dynamic-width speculative beam decoding (DSBD). Specifically, we first introduce a novel draft and verification scheme that generates multiple sequences following the large model's distribution based on beam sampling trajectories from the small model. Then, we introduce an adaptive mechanism to dynamically tune the number of beams based on the context, optimizing efficiency and effectiveness. Besides, we extend tree-based parallel verification to handle multiple trees simultaneously, accelerating the verification process. Finally, we illustrate a simple modification to our algorithm to mitigate the memory overhead of beam sampling...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.21030v2">Standards for Belief Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to demonstrate remarkable abilities across various domains, computer scientists are developing methods to understand their cognitive processes, particularly concerning how (and if) LLMs internally represent their beliefs about the world. However, this field currently lacks a unified theoretical foundation to underpin the study of belief in LLMs. This article begins filling this gap by proposing adequacy conditions for a representation in an LLM to count as belief-like. We argue that, while the project of belief measurement in LLMs shares striking features with belief measurement as carried out in decision theory and formal epistemology, it also differs in ways that should change how we measure belief. Thus, drawing from insights in philosophy and contemporary practices of machine learning, we establish four criteria that balance theoretical considerations with practical constraints. Our proposed criteria include accuracy, coherence, uniformity, and use, which together help lay the groundwork for a comprehensive understanding of belief representation in LLMs. We draw on empirical work showing the limitations of using various criteria in isolation to identify belief representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11495v1">V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 A benchmark for Video Spatio-Temporal Reasoning
    </div>
    <details class="paper-abstract">
      Human processes video reasoning in a sequential spatio-temporal reasoning logic, we first identify the relevant frames ("when") and then analyse the spatial relationships ("where") between key objects, and finally leverage these relationships to draw inferences ("what"). However, can Video Large Language Models (Video-LLMs) also "reason through a sequential spatio-temporal logic" in videos? Existing Video-LLM benchmarks primarily focus on assessing object presence, neglecting relational reasoning. Consequently, it is difficult to measure whether a model truly comprehends object interactions (actions/events) in videos or merely relies on pre-trained "memory" of co-occurrences as biases in generating answers. In this work, we introduce a Video Spatio-Temporal Reasoning (V-STaR) benchmark to address these shortcomings. The key idea is to decompose video understanding into a Reverse Spatio-Temporal Reasoning (RSTR) task that simultaneously evaluates what objects are present, when events occur, and where they are located while capturing the underlying Chain-of-thought (CoT) logic. To support this evaluation, we construct a dataset to elicit the spatial-temporal reasoning process of Video-LLMs. It contains coarse-to-fine CoT questions generated by a semi-automated GPT-4-powered pipeline, embedding explicit reasoning chains to mimic human cognition. Experiments from 14 Video-LLMs on our V-STaR reveal significant gaps between current Video-LLMs and the needs for robust and consistent spatio-temporal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11458v1">Integrating LLMs in Gamified Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 9 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      In this work, a thorough mathematical framework for incorporating Large Language Models (LLMs) into gamified systems is presented with an emphasis on improving task dynamics, user engagement, and reward systems. Personalized feedback, adaptive learning, and dynamic content creation are all made possible by integrating LLMs and are crucial for improving user engagement and system performance. A simulated environment tests the framework's adaptability and demonstrates its potential for real-world applications in various industries, including business, healthcare, and education. The findings demonstrate how LLMs can offer customized experiences that raise system effectiveness and user retention. This study also examines the difficulties this framework aims to solve, highlighting its importance in maximizing involvement and encouraging sustained behavioral change in a range of sectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11441v1">D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Recent advancements in instruction tuning for large language models (LLMs) suggest that a small, high-quality dataset can significantly equip LLMs with instruction-following capabilities, outperforming large datasets often burdened by quality and redundancy issues. However, the challenge lies in automatically identifying valuable subsets from large datasets to boost both the effectiveness and efficiency of instruction tuning. In this paper, we first establish data selection criteria based on three distinct aspects of data value: diversity, difficulty, and dependability, and then propose the D3 method comprising two key steps of scoring and selection. Specifically, in the scoring step, we define the diversity function to measure sample distinctiveness and introduce the uncertainty-based prediction difficulty to evaluate sample difficulty by mitigating the interference of context-oriented generation diversity. Additionally, we integrate an external LLM for dependability assessment. In the selection step, we formulate the D3 weighted coreset objective, which jointly optimizes three aspects of data value to solve for the most valuable subset. The two steps of D3 can iterate multiple rounds, incorporating feedback to refine the selection focus adaptively. Experiments on three datasets demonstrate the effectiveness of D3 in endowing LLMs with competitive or even superior instruction-following capabilities using less than 10% of the entire dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19209v3">VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 CVPR 2025; First three authors contributed equally; Project page: https://videotree2024.github.io/
    </div>
    <details class="paper-abstract">
      Long-form video understanding is complicated by the high redundancy of video data and the abundance of query-irrelevant information. To tackle these challenges, we propose VideoTree, a training-free framework which builds a query-adaptive and hierarchical video representation for LLM reasoning over long-form videos. First, VideoTree extracts query-relevant information from the input video through an iterative process, progressively refining the selection of keyframes based on their relevance to the query. Furthermore, VideoTree leverages the inherent hierarchical structure of long video data, which is often overlooked by existing LLM-based methods. Specifically, we incorporate multi-granularity information into a tree-based representation, allowing VideoTree to extract query-relevant details from long videos in a coarse-to-fine manner. This enables the model to effectively handle a wide range of video queries with varying levels of detail. Finally, VideoTree aggregates the hierarchical query-relevant information within the tree structure and feeds it into an LLM reasoning model to answer the query. Our experiments show that our method improves both reasoning accuracy and efficiency. Specifically, VideoTree outperforms existing training-free approaches on EgoSchema and NExT-QA with less inference time, achieving 61.1% and 75.6% accuracy on the test set without additional video-specific training. Moreover, on the long split of Video-MME (average 44 minutes), VideoTree achieves better performance than GPT-4V and many other MLLMs that were extensively trained on video data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v2">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11315v1">MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 The code and models are available https://github.com/JeongHun0716/MMS-LLaMA
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) achieves robust speech recognition in noisy environments by combining auditory and visual information. However, recent Large Language Model (LLM) based AVSR systems incur high computational costs due to the high temporal resolution of audio-visual speech processed by LLMs. In this work, we introduce an efficient multimodal speech LLM framework that minimizes token length while preserving essential linguistic content. Our approach employs an early av-fusion module for streamlined feature integration, an audio-visual speech Q-Former that dynamically allocates tokens based on input duration, and a refined query allocation strategy with a speech rate predictor to adjust token allocation according to speaking speed of each audio sample. Extensive experiments on the LRS3 dataset show that our method achieves state-of-the-art performance with a WER of 0.74% while using only 3.5 tokens per second. Moreover, our approach not only reduces token usage by 86% compared to the previous multimodal speech LLM framework, but also improves computational efficiency by reducing FLOPs by 35.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11256v1">Line of Duty: Evaluating LLM Self-Knowledge via Consistency in Feasibility Boundaries</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 14 pages, 8 figures, Accepted to the 5th TrustNLP Workshop at NAACL 2025
    </div>
    <details class="paper-abstract">
      As LLMs grow more powerful, their most profound achievement may be recognising when to say "I don't know". Existing studies on LLM self-knowledge have been largely constrained by human-defined notions of feasibility, often neglecting the reasons behind unanswerability by LLMs and failing to study deficient types of self-knowledge. This study aims to obtain intrinsic insights into different types of LLM self-knowledge with a novel methodology: allowing them the flexibility to set their own feasibility boundaries and then analysing the consistency of these limits. We find that even frontier models like GPT-4o and Mistral Large are not sure of their own capabilities more than 80% of the time, highlighting a significant lack of trustworthiness in responses. Our analysis of confidence balance in LLMs indicates that models swing between overconfidence and conservatism in feasibility boundaries depending on task categories and that the most significant self-knowledge weaknesses lie in temporal awareness and contextual understanding. These difficulties in contextual comprehension additionally lead models to question their operational boundaries, resulting in considerable confusion within the self-knowledge of LLMs. We make our code and results available publicly at https://github.com/knowledge-verse-ai/LLM-Self_Knowledge_Eval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11237v1">Collaboration is all you need: LLM Assisted Safe Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      This paper introduces UniTranslator, a visionary framework that re-imagines code translation as a collaborative endeavor among multiple, compact LLMs. By orchestrating the interaction of specialized agents, each focused on different aspects of the translation process and grounded in a deep understanding of programming concepts, UniTranslator achieves a level of accuracy and efficiency that rivals larger, monolithic models. Our preliminary evaluation demonstrates the potential of UniTranslator to overcome the limitations of existing approaches and unlock the power of smaller LLMs for complex code translation tasks. We explore the effectiveness of this dynamic multi-agent paradigm in handling diverse language pairs, including low-resource languages, and in mitigating common issues such as code artifacts and hallucinations through the use of Natural Language Inference (NLI) grounding and iterative feedback mechanisms
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11232v1">PrivacyScalpel: Enhancing LLM Privacy via Interpretable Feature Intervention with Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing but also pose significant privacy risks by memorizing and leaking Personally Identifiable Information (PII). Existing mitigation strategies, such as differential privacy and neuron-level interventions, often degrade model utility or fail to effectively prevent leakage. To address this challenge, we introduce PrivacyScalpel, a novel privacy-preserving framework that leverages LLM interpretability techniques to identify and mitigate PII leakage while maintaining performance. PrivacyScalpel comprises three key steps: (1) Feature Probing, which identifies layers in the model that encode PII-rich representations, (2) Sparse Autoencoding, where a k-Sparse Autoencoder (k-SAE) disentangles and isolates privacy-sensitive features, and (3) Feature-Level Interventions, which employ targeted ablation and vector steering to suppress PII leakage. Our empirical evaluation on Gemma2-2b and Llama2-7b, fine-tuned on the Enron dataset, shows that PrivacyScalpel significantly reduces email leakage from 5.15\% to as low as 0.0\%, while maintaining over 99.4\% of the original model's utility. Notably, our method outperforms neuron-level interventions in privacy-utility trade-offs, demonstrating that acting on sparse, monosemantic features is more effective than manipulating polysemantic neurons. Beyond improving LLM privacy, our approach offers insights into the mechanisms underlying PII memorization, contributing to the broader field of model interpretability and secure AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17174v2">CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Causal Significance and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Chain-based reasoning methods like chain of thought (CoT) play a rising role in solving reasoning tasks for large language models (LLMs). However, the causal illusions between \textit{a step of reasoning} and \textit{corresponding state transitions} are becoming a significant obstacle to advancing LLMs' reasoning capabilities, especially in long-range reasoning tasks. This paper proposes a non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE). We customize LLM's loss function utilizing treatment effect assessments to enhance its reasoning ability from two aspects: causal significance and consistency. This ensures that the model captures essential causal relationships and maintains robust and consistent performance across various scenarios. Additionally, we transform the reasoning process from the cascading multiple one-step reasoning commonly used in Chain-Based methods, like CoT, to a causal-enhanced method that outputs the entire reasoning process in one go, further improving the model's reasoning efficiency. Extensive experiments show that our method improves both the reasoning success rate and speed. These improvements further demonstrate that non-chain-based methods can also aid LLMs in completing reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11205v1">LLaVA-MLB: Mitigating and Leveraging Attention Bias for Training-Free Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Training-free video large language models (LLMs) leverage pretrained Image LLMs to process video content without the need for further training. A key challenge in such approaches is the difficulty of retaining essential visual and temporal information, constrained by the token limits in Image LLMs. To address this, we propose a two-stage method for selecting query-relevant tokens based on the LLM attention scores: compressing the video sequence and then expanding the sequence. However, during the compression stage, Image LLMs often exhibit a positional attention bias in video sequences, where attention is overly concentrated on later frames, causing early-frame information to be underutilized. To alleviate this attention bias during sequence compression, we propose Gridded Attention Pooling for preserving spatiotemporal structure. Additionally, we introduce Visual Summarization Tail to effectively utilize this bias, facilitating overall video understanding during sequence expansion. In this way, our method effectively Mitigates and Leverages attention Bias (LLaVA-MLB), enabling the frozen Image LLM for detailed video understanding. Experiments on several benchmarks demonstrate that our approach outperforms state-of-the-art methods, achieving superior performance in both efficiency and accuracy. Our code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11164v1">Towards Extreme Pruning of LLMs with Plug-and-Play Mixed Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      N:M structured pruning is essential for large language models (LLMs) because it can remove less important network weights and reduce the memory and computation requirements. Existing pruning methods mainly focus on designing metrics to measure the importance of network components to guide pruning. Apart from the impact of these metrics, we observe that different layers have different sensitivities over the network performance. Thus, we propose an efficient method based on the trace of Fisher Information Matrix (FIM) to quantitatively measure and verify the different sensitivities across layers. Based on this, we propose Mixed Sparsity Pruning (MSP) which uses a pruning-oriented evolutionary algorithm (EA) to determine the optimal sparsity levels for different layers. To guarantee fast convergence and achieve promising performance, we utilize efficient FIM-inspired layer-wise sensitivity to initialize the population of EA. In addition, our MSP can work as a plug-and-play module, ready to be integrated into existing pruning methods. Extensive experiments on LLaMA and LLaMA-2 on language modeling and zero-shot tasks demonstrate our superior performance. In particular, in extreme pruning ratio (e.g. 75%), our method significantly outperforms existing methods in terms of perplexity (PPL) by orders of magnitude (Figure 1).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09501v2">ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Recent research on Reasoning of Large Language Models (LLMs) has sought to further enhance their performance by integrating meta-thinking -- enabling models to monitor, evaluate, and control their reasoning processes for more adaptive and effective problem-solving. However, current single-agent work lacks a specialized design for acquiring meta-thinking, resulting in low efficacy. To address this challenge, we introduce Reinforced Meta-thinking Agents (ReMA), a novel framework that leverages Multi-Agent Reinforcement Learning (MARL) to elicit meta-thinking behaviors, encouraging LLMs to think about thinking. ReMA decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions. Through iterative reinforcement learning with aligned objectives, these agents explore and learn collaboration, leading to improved generalization and robustness. Experimental results demonstrate that ReMA outperforms single-agent RL baselines on complex reasoning tasks, including competitive-level mathematical benchmarks and LLM-as-a-Judge benchmarks. Comprehensive ablation studies further illustrate the evolving dynamics of each distinct agent, providing valuable insights into how the meta-thinking reasoning process enhances the reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11082v1">LLMs are Bug Replicators: An Empirical Study on LLMs' Capability in Completing Bug-prone Code</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance in code completion. However, the training data used to develop these models often contain a significant amount of buggy code. Yet, it remains unclear to what extent these buggy instances influence LLMs' performance when tackling bug-prone code completion tasks. To fill this gap, this paper presents the first empirical study evaluating the performance of LLMs in completing bug-prone code. Through extensive experiments on 7 LLMs and the Defects4J dataset, we analyze LLMs' accuracy, robustness, and limitations in this challenging context. Our experimental results show that completing bug-prone code is significantly more challenging for LLMs than completing normal code. Notably, in bug-prone tasks, the likelihood of LLMs generating correct code is nearly the same as generating buggy code, and it is substantially lower than in normal code completion tasks (e.g., 12.27% vs. 29.85% for GPT-4). To our surprise, 44.44% of the bugs LLMs make are completely identical to the pre-fix version, indicating that LLMs have been seriously biased by historical bugs when completing code. Additionally, we investigate the effectiveness of existing post-processing techniques and find that while they can improve consistency, they do not significantly reduce error rates in bug-prone code scenarios. Our research highlights the limitations of current LLMs in handling bug-prone code and underscores the need for improved models and post-processing strategies to enhance code completion accuracy in real-world development environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03279v4">Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 EMNLP 2024 main
    </div>
    <details class="paper-abstract">
      Model editing aims to correct outdated or erroneous knowledge in large language models (LLMs) without the need for costly retraining. Lifelong model editing is the most challenging task that caters to the continuous editing requirements of LLMs. Prior works primarily focus on single or batch editing; nevertheless, these methods fall short in lifelong editing scenarios due to catastrophic knowledge forgetting and the degradation of model performance. Although retrieval-based methods alleviate these issues, they are impeded by slow and cumbersome processes of integrating the retrieved knowledge into the model. In this work, we introduce RECIPE, a RetriEval-augmented ContInuous Prompt lEarning method, to boost editing efficacy and inference efficiency in lifelong learning. RECIPE first converts knowledge statements into short and informative continuous prompts, prefixed to the LLM's input query embedding, to efficiently refine the response grounded on the knowledge. It further integrates the Knowledge Sentinel (KS) that acts as an intermediary to calculate a dynamic threshold, determining whether the retrieval repository contains relevant knowledge. Our retriever and prompt encoder are jointly trained to achieve editing properties, i.e., reliability, generality, and locality. In our experiments, RECIPE is assessed extensively across multiple LLMs and editing datasets, where it achieves superior editing performance. RECIPE also demonstrates its capability to maintain the overall performance of LLMs alongside showcasing fast editing and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11023v1">Beyond A Single AI Cluster: A Survey of Decentralized LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has revolutionized AI development, yet their training demands computational resources beyond a single cluster or even datacenter, limiting accessibility to large organizations. Decentralized training has emerged as a promising paradigm to leverage dispersed resources across clusters, datacenters, and global regions, democratizing LLM development for broader communities. As the first comprehensive exploration of this emerging field, we present decentralized LLM training as a resource-driven paradigm and categorize it into community-driven and organizational approaches. Furthermore, our in-depth analysis clarifies decentralized LLM training, including: (1) position with related domain concepts comparison, (2) decentralized resource development trends, and (3) recent advances with discussion under a novel taxonomy. We also provide up-to-date case studies and explore future directions, contributing to the evolution of decentralized LLM training research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11018v1">An LLM's Attempts to Adapt to Diverse Software Engineers' Problem-Solving Styles: More Inclusive & Equitable?</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Software engineers use code-fluent large language models (LLMs) to help explain unfamiliar code, yet LLM explanations are not adapted to engineers' diverse problem-solving needs. We prompted an LLM to adapt to five problem-solving style types from an inclusive design method, the Gender Inclusiveness Magnifier (GenderMag). We ran a user study with software engineers to examine the impact of explanation adaptations on software engineers' perceptions, both for explanations which matched and mismatched engineers' problem-solving styles. We found that explanations were more frequently beneficial when they matched problem-solving style, but not every matching adaptation was equally beneficial; in some instances, diverse engineers found as much (or more) benefit from mismatched adaptations. Through an equity and inclusivity lens, our work highlights the benefits of having an LLM adapt its explanations to match engineers' diverse problem-solving style values, the potential harms when matched adaptations were not perceived well by engineers, and a comparison of how matching and mismatching LLM adaptations impacted diverse engineers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10990v1">Statistical Impossibility and Possibility of Aligning LLMs with Human Preferences: From Condorcet Paradox to Nash Equilibrium</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with diverse human preferences is critical for ensuring fairness and informed outcomes when deploying these models for decision-making. In this paper, we seek to uncover fundamental statistical limits concerning aligning LLMs with human preferences, with a focus on the probabilistic representation of human preferences and the preservation of diverse preferences in aligned LLMs. We first show that human preferences can be represented by a reward model if and only if the preference among LLM-generated responses is free of any Condorcet cycle. Moreover, we prove that Condorcet cycles exist with probability converging to one exponentially fast under a probabilistic preference model, thereby demonstrating the impossibility of fully aligning human preferences using reward-based approaches such as reinforcement learning from human feedback. Next, we explore the conditions under which LLMs would employ mixed strategies -- meaning they do not collapse to a single response -- when aligned in the limit using a non-reward-based approach, such as Nash learning from human feedback (NLHF). We identify a necessary and sufficient condition for mixed strategies: the absence of a response that is preferred over all others by a majority. As a blessing, we prove that this condition holds with high probability under the probabilistic preference model, thereby highlighting the statistical possibility of preserving minority preferences without explicit regularization in aligning LLMs. Finally, we leverage insights from our statistical results to design a novel, computationally efficient algorithm for finding Nash equilibria in aligning LLMs with NLHF. Our experiments show that Llama-3.2-1B, aligned with our algorithm, achieves a win rate of 60.55\% against the base model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10968v1">Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05786v2">FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 9 pages, 3 figures, 2 tables and 2 algorithms
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06917v2">Combinatorial Optimization via LLM-driven Iterated Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      We present a novel way to integrate flexible, context-dependent constraints into combinatorial optimization by leveraging Large Language Models (LLMs) alongside traditional algorithms. Although LLMs excel at interpreting nuanced, locally specified requirements, they struggle with enforcing global combinatorial feasibility. To bridge this gap, we propose an iterated fine-tuning framework where algorithmic feedback progressively refines the LLM's output distribution. Interpreting this as simulated annealing, we introduce a formal model based on a "coarse learnability" assumption, providing sample complexity bounds for convergence. Empirical evaluations on scheduling, graph connectivity, and clustering tasks demonstrate that our framework balances the flexibility of locally expressed constraints with rigorous global optimization more effectively compared to baseline sampling methods. Our results highlight a promising direction for hybrid AI-driven combinatorial reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11911v1">LAG-MMLU: Benchmarking Frontier LLM Understanding in Latvian and Giriama</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 Accepted in NoDaLiDa/Baltic-HLT 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) rapidly advance, evaluating their performance is critical. LLMs are trained on multilingual data, but their reasoning abilities are mainly evaluated using English datasets. Hence, robust evaluation frameworks are needed using high-quality non-English datasets, especially low-resource languages (LRLs). This study evaluates eight state-of-the-art (SOTA) LLMs on Latvian and Giriama using a Massive Multitask Language Understanding (MMLU) subset curated with native speakers for linguistic and cultural relevance. Giriama is benchmarked for the first time. Our evaluation shows that OpenAI's o1 model outperforms others across all languages, scoring 92.8\% in English, 88.8\% in Latvian, and 70.8\% in Giriama on 0-shot tasks. Mistral-large (35.6\%) and Llama-70B IT (41\%) have weak performance, on both Latvian and Giriama. Our results underscore the need for localized benchmarks and human evaluations in advancing cultural AI contextualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11898v1">LLMs for Translation: Historical, Low-Resourced Languages and Contemporary AI Models</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 Accepted to LaTeCH-CLfL 2025, held in conjunction with NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable adaptability in performing various tasks, including machine translation (MT), without explicit training. Models such as OpenAI's GPT-4 and Google's Gemini are frequently evaluated on translation benchmarks and utilized as translation tools due to their high performance. This paper examines Gemini's performance in translating an 18th-century Ottoman Turkish manuscript, Prisoner of the Infidels: The Memoirs of Osman Agha of Timisoara, into English. The manuscript recounts the experiences of Osman Agha, an Ottoman subject who spent 11 years as a prisoner of war in Austria, and includes his accounts of warfare and violence. Our analysis reveals that Gemini's safety mechanisms flagged between 14 and 23 percent of the manuscript as harmful, resulting in untranslated passages. These safety settings, while effective in mitigating potential harm, hinder the model's ability to provide complete and accurate translations of historical texts. Through real historical examples, this study highlights the inherent challenges and limitations of current LLM safety implementations in the handling of sensitive and context-rich materials. These real-world instances underscore potential failures of LLMs in contemporary translation scenarios, where accurate and comprehensive translations are crucial-for example, translating the accounts of modern victims of war for legal proceedings or humanitarian documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11881v1">GPT's Devastated and LLaMA's Content: Emotion Representation Alignment in LLMs for Keyword-based Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      In controlled text generation using large language models (LLMs), gaps arise between the language model's interpretation and human expectations. We look at the problem of controlling emotions in keyword-based sentence generation for both GPT-4 and LLaMA-3. We selected four emotion representations: Words, Valence-Arousal-Dominance (VAD) dimensions expressed in both Lexical and Numeric forms, and Emojis. Our human evaluation looked at the Human-LLM alignment for each representation, as well as the accuracy and realism of the generated sentences. While representations like VAD break emotions into easy-to-compute components, our findings show that people agree more with how LLMs generate when conditioned on English words (e.g., "angry") rather than VAD scales. This difference is especially visible when comparing Numeric VAD to words. However, we found that converting the originally-numeric VAD scales to Lexical scales (e.g., +4.0 becomes "High") dramatically improved agreement. Furthermore, the perception of how much a generated sentence conveys an emotion is highly dependent on the LLM, representation type, and which emotion it is.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11858v1">OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16266v2">REBEL: Rule-based and Experience-enhanced Learning with LLMs for Initial Task Allocation in Multi-Human Multi-Robot Teaming</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Multi-human multi-robot teams are increasingly recognized for their efficiency in executing large-scale, complex tasks by integrating heterogeneous yet potentially synergistic humans and robots. However, this inherent heterogeneity presents significant challenges in teaming, necessitating efficient initial task allocation (ITA) strategies that optimally form complementary human-robot pairs or collaborative chains and establish well-matched task distributions. While current learning-based methods demonstrate promising performance, they often incur high computational costs and lack the flexibility to incorporate user preferences in multi-objective optimization (MOO) or adapt to last-minute changes in dynamic real-world environments. To address these limitations, we propose REBEL, an LLM-based ITA framework that integrates rule-based and experience-enhanced learning to enhance LLM reasoning capabilities and improve in-context adaptability to MOO and situational changes. Extensive experiments validate the effectiveness of REBEL in both single-objective and multi-objective scenarios, demonstrating superior alignment with user preferences and enhanced situational awareness to handle unexpected team composition changes. Additionally, we show that REBEL can complement pre-trained ITA policies, further boosting situational adaptability and overall team performance. Website at https://sites.google.com/view/ita-rebel .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11827v1">Bridging the LLM Accessibility Divide? Performance, Fairness, and Cost of Closed versus Open LLMs for Automated Essay Scoring</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Closed large language models (LLMs) such as GPT-4 have set state-of-the-art results across a number of NLP tasks and have become central to NLP and machine learning (ML)-driven solutions. Closed LLMs' performance and wide adoption has sparked considerable debate about their accessibility in terms of availability, cost, and transparency. In this study, we perform a rigorous comparative analysis of nine leading LLMs, spanning closed, open, and open-source LLM ecosystems, across text assessment and generation tasks related to automated essay scoring. Our findings reveal that for few-shot learning-based assessment of human generated essays, open LLMs such as Llama 3 and Qwen2.5 perform comparably to GPT-4 in terms of predictive performance, with no significant differences in disparate impact scores when considering age- or race-related fairness. Moreover, Llama 3 offers a substantial cost advantage, being up to 37 times more cost-efficient than GPT-4. For generative tasks, we find that essays generated by top open LLMs are comparable to closed LLMs in terms of their semantic composition/embeddings and ML assessed scores. Our findings challenge the dominance of closed LLMs and highlight the democratizing potential of open LLMs, suggesting they can effectively bridge accessibility divides while maintaining competitive performance and fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11733v1">LLM Agents for Education: Advances and Applications</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have demonstrated remarkable capabilities in automating tasks and driving innovation across diverse educational applications. In this survey, we provide a systematic review of state-of-the-art research on LLM agents in education, categorizing them into two broad classes: (1) \emph{Pedagogical Agents}, which focus on automating complex pedagogical tasks to support both teachers and students; and (2) \emph{Domain-Specific Educational Agents}, which are tailored for specialized fields such as science education, language learning, and professional development. We comprehensively examine the technological advancements underlying these LLM agents, including key datasets, benchmarks, and algorithmic frameworks that drive their effectiveness. Furthermore, we discuss critical challenges such as privacy, bias and fairness concerns, hallucination mitigation, and integration with existing educational ecosystems. This survey aims to provide a comprehensive technological overview of LLM agents for education, fostering further research and collaboration to enhance their impact for the greater good of learners and educators alike.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13518v1">Examples as the Prompt: A Scalable Approach for Efficient LLM Adaptation in E-Commerce</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Prompting LLMs offers an efficient way to guide output generation without explicit model training. In the e-commerce domain, prompting-based applications are widely used for tasks such as query understanding, recommender systems, and customer support. However, adapting LLMs to different tasks often requires extensive prompt engineering by domain experts, along with frequent updates to align with evolving business needs. Additionally, crafting fully unbiased natural language prompts remains a challenge for humans. To address these challenges, we propose a novel framework, Examples as the Prompt (EaP) which leverages labeled data to enhance prompts. Specifically, EaP automatically selects the most representative examples to maximize the few-shot capability of LLMs. It is efficient due to its unsupervised example selection and adaptive to potential data distribution shifts. We validate EaP on four real-world production use cases, demonstrating that it achieves comparable or even superior performance comparing to hand-crafted prompts designed by domain experts. Additionally, we introduce EaP_lite, which entirely replaces the natural language components of prompts with labeled examples. EaP_lite improves LLM inference speed by up to 70% without compromising performance. Latest online A/B test shows that using EaP and EaP_lite for data labeling can bring significant composite revenue gain by 0.06%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13517v1">CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 Accepted at ICLR 2025 main conference
    </div>
    <details class="paper-abstract">
      Scientific problem-solving involves synthesizing information while applying expert knowledge. We introduce CURIE, a scientific long-Context Understanding,Reasoning and Information Extraction benchmark to measure the potential of Large Language Models (LLMs) in scientific problem-solving and assisting scientists in realistic workflows. This benchmark introduces ten challenging tasks with a total of 580 problems and solution pairs curated by experts in six disciplines - materials science, condensed matter physics, quantum computing, geospatial analysis, biodiversity, and proteins - covering both experimental and theoretical work-flows in science. We evaluate a range of closed and open LLMs on tasks in CURIE which requires domain expertise, comprehension of long in-context information,and multi-step reasoning. While Gemini Flash 2.0 and Claude-3 show consistent high comprehension across domains, the popular GPT-4o and command-R+ fail dramatically on protein sequencing tasks. With the best performance at 32% there is much room for improvement for all models. We hope that insights gained from CURIE can guide the future development of LLMs in sciences. Evaluation code and data are in https://github.com/google/curie
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13514v1">RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning through RAG and Incremental Knowledge Graph Learning Integration</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      This paper presents RAG-KG-IL, a novel multi-agent hybrid framework designed to enhance the reasoning capabilities of Large Language Models (LLMs) by integrating Retrieval-Augmented Generation (RAG) and Knowledge Graphs (KGs) with an Incremental Learning (IL) approach. Despite recent advancements, LLMs still face significant challenges in reasoning with structured data, handling dynamic knowledge evolution, and mitigating hallucinations, particularly in mission-critical domains. Our proposed RAG-KG-IL framework addresses these limitations by employing a multi-agent architecture that enables continuous knowledge updates, integrates structured knowledge, and incorporates autonomous agents for enhanced explainability and reasoning. The framework utilizes RAG to ensure the generated responses are grounded in verifiable information, while KGs provide structured domain knowledge for improved consistency and depth of understanding. The Incremental Learning approach allows for dynamic updates to the knowledge base without full retraining, significantly reducing computational overhead and improving the model's adaptability. We evaluate the framework using real-world case studies involving health-related queries, comparing it to state-of-the-art models like GPT-4o and a RAG-only baseline. Experimental results demonstrate that our approach significantly reduces hallucination rates and improves answer completeness and reasoning accuracy. The results underscore the potential of combining RAG, KGs, and multi-agent systems to create intelligent, adaptable systems capable of real-time knowledge integration and reasoning in complex domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13510v1">Prompt Sentiment: The Catalyst for LLM Change</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has revolutionized natural language processing (NLP), yet the influence of prompt sentiment, a latent affective characteristic of input text, remains underexplored. This study systematically examines how sentiment variations in prompts affect LLM-generated outputs in terms of coherence, factuality, and bias. Leveraging both lexicon-based and transformer-based sentiment analysis methods, we categorize prompts and evaluate responses from five leading LLMs: Claude, DeepSeek, GPT-4, Gemini, and LLaMA. Our analysis spans six AI-driven applications, including content generation, conversational AI, legal and financial analysis, healthcare AI, creative writing, and technical documentation. By transforming prompts, we assess their impact on output quality. Our findings reveal that prompt sentiment significantly influences model responses, with negative prompts often reducing factual accuracy and amplifying bias, while positive prompts tend to increase verbosity and sentiment propagation. These results highlight the importance of sentiment-aware prompt engineering for ensuring fair and reliable AI-generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10620v1">From TOWER to SPIRE: Adding the Speech Modality to a Text-Only LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance and generalization capabilities across multiple languages and tasks, making them very attractive targets for multi-modality integration (e.g., images or speech). In this work, we extend an existing LLM to the speech modality via speech discretization and continued pre-training. In particular, we are interested in multilingual LLMs, such as TOWER, as their pre-training setting allows us to treat discretized speech input as an additional translation language. The resulting open-source model, SPIRE, is able to transcribe and translate English speech input while maintaining TOWER's original performance on translation-related tasks, showcasing that discretized speech input integration as an additional language is feasible during LLM adaptation. We make our code and models available to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10515v1">Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 18 pages, 7 figures, 3 tables, code: https://github.com/mainlp/discourse_probes
    </div>
    <details class="paper-abstract">
      Discourse understanding is essential for many NLP tasks, yet most existing work remains constrained by framework-dependent discourse representations. This work investigates whether large language models (LLMs) capture discourse knowledge that generalizes across languages and frameworks. We address this question along two dimensions: (1) developing a unified discourse relation label set to facilitate cross-lingual and cross-framework discourse analysis, and (2) probing LLMs to assess whether they encode generalizable discourse abstractions. Using multilingual discourse relation classification as a testbed, we examine a comprehensive set of 23 LLMs of varying sizes and multilingual capabilities. Our results show that LLMs, especially those with multilingual training corpora, can generalize discourse information across languages and frameworks. Further layer-wise analyses reveal that language generalization at the discourse level is most salient in the intermediate layers. Lastly, our error analysis provides an account of challenging relation classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13640v2">Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      LLM self-evaluation relies on the LLM's own ability to estimate response correctness, which can greatly improve its deployment reliability. In this research track, we propose the Chain-of-Embedding (CoE) in the latent space to enable LLMs to perform output-free self-evaluation. CoE consists of all progressive hidden states produced during the inference time, which can be treated as the latent thinking path of LLMs. We find that when LLMs respond correctly and incorrectly, their CoE features differ, these discrepancies assist us in estimating LLM response correctness. Experiments in four diverse domains and seven LLMs fully demonstrate the effectiveness of our method. Meanwhile, its label-free design intent without any training and millisecond-level computational cost ensures real-time feedback in large-scale scenarios. More importantly, we provide interesting insights into LLM response correctness from the perspective of hidden state changes inside LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10486v1">LLMs in Disease Diagnosis: A Comparative Study of DeepSeek-R1 and O3 Mini Across Chronic Health Conditions</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing medical diagnostics by enhancing both disease classification and clinical decision-making. In this study, we evaluate the performance of two LLM- based diagnostic tools, DeepSeek R1 and O3 Mini, using a structured dataset of symptoms and diagnoses. We assessed their predictive accuracy at both the disease and category levels, as well as the reliability of their confidence scores. DeepSeek R1 achieved a disease-level accuracy of 76% and an overall accuracy of 82%, outperforming O3 Mini, which attained 72% and 75% respectively. Notably, DeepSeek R1 demonstrated exceptional performance in Mental Health, Neurological Disorders, and Oncology, where it reached 100% accuracy, while O3 Mini excelled in Autoimmune Disease classification with 100% accuracy. Both models, however, struggled with Respiratory Disease classification, recording accuracies of only 40% for DeepSeek R1 and 20% for O3 Mini. Additionally, the analysis of confidence scores revealed that DeepSeek R1 provided high-confidence predictions in 92% of cases, compared to 68% for O3 Mini. Ethical considerations regarding bias, model interpretability, and data privacy are also discussed to ensure the responsible integration of LLMs into clinical practice. Overall, our findings offer valuable insights into the strengths and limitations of LLM-based diagnostic systems and provide a roadmap for future enhancements in AI-driven healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10377v1">SPPO:Efficient Long-sequence LLM Training via Adaptive Sequence Pipeline Parallel Offloading</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities, driving advancements in real-world applications. However, training LLMs on increasingly long input sequences imposes significant challenges due to high GPU memory and computational demands. Existing solutions face two key limitations: (1) memory reduction techniques, such as activation recomputation and CPU offloading, compromise training efficiency; (2) distributed parallelism strategies require excessive GPU resources, limiting the scalability of input sequence length. To address these gaps, we propose Adaptive Sequence Pipeline Parallel Offloading (SPPO), a novel LLM training framework that optimizes memory and computational resource efficiency for long-sequence training. SPPO introduces adaptive offloading, leveraging sequence-aware offloading, and two-level activation management to reduce GPU memory consumption without degrading the training efficiency. Additionally, SPPO develops an adaptive pipeline scheduling approach with a heuristic solver and multiplexed sequence partitioning to improve computational resource efficiency. Experimental results demonstrate that SPPO achieves up to 3.38x throughput improvement over Megatron-LM and DeepSpeed, realizing efficient training of a 7B LLM with sequence lengths of up to 4M tokens on only 128 A100 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10367v1">G-Boost: Boosting Private SLMs with General LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Due to the limited computational resources, most Large Language Models (LLMs) developers can only fine-tune Small Language Models (SLMs) on their own data. These private SLMs typically have limited effectiveness. To boost the performance of private SLMs, this paper proposes to ask general LLMs for help. The general LLMs can be APIs or larger LLMs whose inference cost the developers can afford. Specifically, we propose the G-Boost framework where a private SLM adaptively performs collaborative inference with a general LLM under the guide of process reward. Experiments demonstrate that our framework can significantly boost the performance of private SLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04070v7">PAD: Personalized Alignment of LLMs at Decoding-Time</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Aligning with personalized preferences, which vary significantly across cultural, educational, and political differences, poses a significant challenge due to the computational costs and data demands of traditional alignment methods. In response, this paper presents Personalized Alignment at Decoding-time (PAD), a novel framework designed to align LLM outputs with diverse personalized preferences during the inference phase, eliminating the need for additional training. By introducing a unique personalized reward modeling strategy, this framework decouples the text generation process from personalized preferences, facilitating the generation of generalizable token-level personalized rewards. The PAD algorithm leverages these rewards to guide the decoding process, dynamically tailoring the base model's predictions to personalized preferences. Extensive experimental results demonstrate that PAD not only outperforms existing training-based alignment methods in terms of aligning with diverse preferences but also shows significant generalizability to preferences unseen during training and scalability across different base models. This work advances the capability of LLMs to meet user needs in real-time applications, presenting a substantial step forward in personalized LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12029v2">KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in various complex tasks, yet they still suffer from hallucinations. Introducing external knowledge, such as knowledge graph, can enhance the LLMs' ability to provide factual answers. LLMs have the ability to interactively explore knowledge graphs. However, most approaches have been affected by insufficient internal knowledge excavation in LLMs, limited generation of trustworthy knowledge reasoning paths, and a vague integration between internal and external knowledge. Therefore, we propose KnowPath, a knowledge-enhanced large model framework driven by the collaboration of internal and external knowledge. It relies on the internal knowledge of the LLM to guide the exploration of interpretable directed subgraphs in external knowledge graphs, better integrating the two knowledge sources for more accurate reasoning. Extensive experiments on multiple real-world datasets confirm the superiority of KnowPath.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10325v1">Collaborative Speculative Inference for Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Speculative inference is a promising paradigm employing small speculative models (SSMs) as drafters to generate draft tokens, which are subsequently verified in parallel by the target large language model (LLM). This approach enhances the efficiency of inference serving by reducing LLM inference latency and costs while preserving generation quality. However, existing speculative methods face critical challenges, including inefficient resource utilization and limited draft acceptance, which constrain their scalability and overall effectiveness. To overcome these obstacles, we present CoSine, a novel speculative inference system that decouples sequential speculative decoding from parallel verification, enabling efficient collaboration among multiple nodes. Specifically, CoSine routes inference requests to specialized drafters based on their expertise and incorporates a confidence-based token fusion mechanism to synthesize outputs from cooperating drafters, ensuring high-quality draft generation. Additionally, CoSine dynamically orchestrates the execution of speculative decoding and verification in a pipelined manner, employing batch scheduling to selectively group requests and adaptive speculation control to minimize idle periods. By optimizing parallel workflows through heterogeneous node collaboration, CoSine balances draft generation and verification throughput in real-time, thereby maximizing resource utilization. Experimental results demonstrate that CoSine achieves superior performance compared to state-of-the-art speculative approaches. Notably, with equivalent resource costs, CoSine achieves up to a 23.2% decrease in latency and a 32.5% increase in throughput compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07384v2">Is My Text in Your AI Model? Gradient-based Membership Inference Test applied to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      This work adapts and studies the gradient-based Membership Inference Test (gMINT) to the classification of text based on LLMs. MINT is a general approach intended to determine if given data was used for training machine learning models, and this work focuses on its application to the domain of Natural Language Processing. Using gradient-based analysis, the MINT model identifies whether particular data samples were included during the language model training phase, addressing growing concerns about data privacy in machine learning. The method was evaluated in seven Transformer-based models and six datasets comprising over 2.5 million sentences, focusing on text classification tasks. Experimental results demonstrate MINTs robustness, achieving AUC scores between 85% and 99%, depending on data size and model architecture. These findings highlight MINTs potential as a scalable and reliable tool for auditing machine learning models, ensuring transparency, safeguarding sensitive data, and fostering ethical compliance in the deployment of AI/NLP technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09672v3">InstructPipe: Generating Visual Blocks Pipelines with Human Instructions and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 CHI 2025
    </div>
    <details class="paper-abstract">
      Visual programming has the potential of providing novice programmers with a low-code experience to build customized processing pipelines. Existing systems typically require users to build pipelines from scratch, implying that novice users are expected to set up and link appropriate nodes from a blank workspace. In this paper, we introduce InstructPipe, an AI assistant for prototyping machine learning (ML) pipelines with text instructions. We contribute two large language model (LLM) modules and a code interpreter as part of our framework. The LLM modules generate pseudocode for a target pipeline, and the interpreter renders the pipeline in the node-graph editor for further human-AI collaboration. Both technical and user evaluation (N=16) shows that InstructPipe empowers users to streamline their ML pipeline workflow, reduce their learning curve, and leverage open-ended commands to spark innovative ideas.
    </details>
</div>
