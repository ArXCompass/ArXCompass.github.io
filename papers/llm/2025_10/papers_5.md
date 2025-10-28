# llm - 2025_10

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
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16686v1">Investigating the Impact of Rationales for LLMs on Natural Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) rationales, which provide step-by-step reasoning to derive final answers, benefit LLMs in both inference and training. Incorporating rationales, either by generating them before answering during inference, or by placing them before or after the original answers during training - significantly improves model performance on mathematical, symbolic and commonsense reasoning tasks. However, most work focuses on the role of rationales in these reasoning tasks, overlooking their potential impact on other important tasks like natural language understanding (NLU) tasks. In this work, we raise the question: Can rationales similarly benefit NLU tasks? To conduct a systematic exploration, we construct NLURC, a comprehensive and high-quality NLU dataset collection with rationales, and develop various rationale-augmented methods. Through exploring the applicability of these methods on NLU tasks using the dataset, we uncover several potentially surprising findings: (1) CoT inference shifts from hindering NLU performance to surpassing direct label prediction as model size grows, indicating a positive correlation. (2) Most rationale-augmented training methods perform worse than label-only training, with one specially designed method consistently achieving improvements. (3) LLMs trained with rationales achieve significant performance gains on unseen NLU tasks, rivaling models ten times their size, while delivering interpretability on par with commercial LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v3">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Published as a conference paper at Neurips 2025
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17913v1">TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted for publication in the proceedings of ICTAI 2025
    </div>
    <details class="paper-abstract">
      Simulating nuanced human social dynamics with Large Language Models (LLMs) remains a significant challenge, particularly in achieving psychological depth and consistent persona behavior crucial for high-fidelity training tools. This paper introduces TACLA (Transactional Analysis Contextual LLM-based Agents), a novel Multi-Agent architecture designed to overcome these limitations. TACLA integrates core principles of Transactional Analysis (TA) by modeling agents as an orchestrated system of distinct Parent, Adult, and Child ego states, each with its own pattern memory. An Orchestrator Agent prioritizes ego state activation based on contextual triggers and an agent's life script, ensuring psychologically authentic responses. Validated in an educational scenario, TACLA demonstrates realistic ego state shifts in Student Agents, effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies. Evaluation shows high conversational credibility and confirms TACLA's capacity to create dynamic, psychologically-grounded social simulations, advancing the development of effective AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17910v1">Interpretability Framework for LLMs in Undergraduate Calculus</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used in education, yet their correctness alone does not capture the quality, reliability, or pedagogical validity of their problem-solving behavior, especially in mathematics, where multistep logic, symbolic reasoning, and conceptual clarity are critical. Conventional evaluation methods largely focus on final answer accuracy and overlook the reasoning process. To address this gap, we introduce a novel interpretability framework for analyzing LLM-generated solutions using undergraduate calculus problems as a representative domain. Our approach combines reasoning flow extraction and decomposing solutions into semantically labeled operations and concepts with prompt ablation analysis to assess input salience and output stability. Using structured metrics such as reasoning complexity, phrase sensitivity, and robustness, we evaluated the model behavior on real Calculus I to III university exams. Our findings revealed that LLMs often produce syntactically fluent yet conceptually flawed solutions, with reasoning patterns sensitive to prompt phrasing and input variation. This framework enables fine-grained diagnosis of reasoning failures, supports curriculum alignment, and informs the design of interpretable AI-assisted feedback tools. This is the first study to offer a structured, quantitative, and pedagogically grounded framework for interpreting LLM reasoning in mathematics education, laying the foundation for the transparent and responsible deployment of AI in STEM learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17904v1">BreakFun: Jailbreaking LLMs via Schema Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17902v1">Activation Manifold Projection: Liberating Task-Specific Behaviors from LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Model (LLM) architectures presents a fundamental challenge: valuable, task-specific behaviors learned through fine-tuning methods like Low-Rank Adaptation (LoRA) are effectively trapped within their source model's architecture, herein referred to architectural lock-in. Existing transfer methods attempt to bridge this gap by aligning the static weight spaces of models, a brittle and indirect approach that relies on tenuous correlations between parameter geometries. This paper introduces a fundamentally different and more direct paradigm: the Cartridge Activation Space Transfer (CAST), a novel framework that liberates LoRA-encoded behaviors by learning a direct, nonlinear mapping between the activation manifolds, the geometric structures formed by the model's internal neuron activations, of two distinct LLM architectures. CAST treats a pre-trained LoRA as a frozen "behavioral kernel." It learns a set of lightweight, bidirectional projection heads that translate the target model's activation stream into the source model's latent space, apply the frozen kernel, and project the result back. This process, trained on a general text corpus without any task-specific data, effectively decouples the learned skill from the source architecture. We demonstrate that CAST enables true "zero-shot" translation of any standard LoRA adapter. Our experiments, including transfers between heterogeneous model families like Llama-2 and Mistral, show that CAST-translated adapters achieve 85-95\% of the performance of a LoRA fully retrained on the target model, quantitatively outperforming current weight-space transfer techniques and establishing a new state-of-the-art in model interoperability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17900v1">Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are entering legal workflows, yet we lack a jurisdiction-specific framework to assess their baseline competence therein. We use India's public legal examinations as a transparent proxy. Our multi-year benchmark assembles objective screens from top national and state exams and evaluates open and frontier LLMs under real-world exam conditions. To probe beyond multiple-choice questions, we also include a lawyer-graded, paired-blinded study of long-form answers from the Supreme Court's Advocate-on-Record exam. This is, to our knowledge, the first exam-grounded, India-specific yardstick for LLM court-readiness released with datasets and protocols. Our work shows that while frontier systems consistently clear historical cutoffs and often match or exceed recent top-scorer bands on objective exams, none surpasses the human topper on long-form reasoning. Grader notes converge on three reliability failure modes: procedural or format compliance, authority or citation discipline, and forum-appropriate voice and structure. These findings delineate where LLMs can assist (checks, cross-statute consistency, statute and precedent lookups) and where human leadership remains essential: forum-specific drafting and filing, procedural and relief strategy, reconciling authorities and exceptions, and ethical, accountable judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17023v1">Enrich and Detect: Video Temporal Grounding with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 ICCV 2025 (Highlights)
    </div>
    <details class="paper-abstract">
      We introduce ED-VTG, a method for fine-grained video temporal grounding utilizing multi-modal large language models. Our approach harnesses the capabilities of multimodal LLMs to jointly process text and video, in order to effectively localize natural language queries in videos through a two-stage process. Rather than being directly grounded, language queries are initially transformed into enriched sentences that incorporate missing details and cues to aid in grounding. In the second stage, these enriched queries are grounded, using a lightweight decoder, which specializes at predicting accurate boundaries conditioned on contextualized representations of the enriched queries. To mitigate noise and reduce the impact of hallucinations, our model is trained with a multiple-instance-learning objective that dynamically selects the optimal version of the query for each training sample. We demonstrate state-of-the-art results across various benchmarks in temporal video grounding and paragraph grounding settings. Experiments reveal that our method significantly outperforms all previously proposed LLM-based temporal grounding approaches and is either superior or comparable to specialized models, while maintaining a clear advantage against them in zero-shot evaluation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17021v1">Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at https://github.com/OPTML-Group/Unlearn-Backdoor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17017v1">SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Code: https://github.com/ZQS1943/SafeSearch
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17015v1">Justitia: Fair and Efficient Scheduling for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      In the era of Large Language Models (LLMs), it has been popular to launch a series of LLM inferences -- we call an LLM application -- to better solve real-world problems. When serving those applications in shared GPU servers, the schedulers are expected to attain fast application completions with guaranteed worst-case performance. However, mainstream LLM schedulers fail to behave well for LLM applications -- due to head-of-line blocking or over-constrained resource allocation. In this paper, we propose to serve LLM applications in a fair and also efficient manner. To this end, we design Justitia, a novel scheduler with three key techniques. First, given that memory is prevalently a bottleneck for mainstream inference frameworks like vLLM, Justitia models the service cost of LLM applications in a memory-centric manner. Meanwhile, it uses a simple neural network model to conduct light-weight and also accurate demand prediction. Moreover, Justitia adopts a virtual-time based fair queuing algorithm to reduce the overall performance with guaranteed worst-case delay. We have implemented Justitia atop vLLM, and experimental results involving diverse LLM applications show that it can substantially enhance the scheduling efficiency with fairness preserved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17013v1">DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v4">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v3">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17002v1">EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Circuit schematics play a crucial role in analog integrated circuit design, serving as the primary medium for human understanding and verification of circuit functionality. While recent large language model (LLM)-based approaches have shown promise in circuit topology generation and device sizing, most rely solely on textual representations such as SPICE netlists, which lack visual interpretability for circuit designers. To address this limitation, we propose EEschematic, an AI agent for automatic analog schematic generation based on a Multimodal Large Language Model (MLLM). EEschematic integrates textual, visual, and symbolic modalities to translate SPICE netlists into schematic diagrams represented in a human-editable format. The framework uses six analog substructure examples for few-shot placement and a Visual Chain-of-Thought (VCoT) strategy to iteratively refine placement and wiring, enhancing schematic clarity and symmetry. Experimental results on representative analog circuits, including a CMOS inverter, a five-transistor operational transconductance amplifier (5T-OTA), and a telescopic cascode amplifier, demonstrate that EEschematic produces schematics with high visual quality and structural correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17001v1">Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) were shown to encode word form variations, such as "walk"->"walked", as linear directions in embedding space. However, standard tokenization algorithms treat these variations as distinct tokens -- filling the size-capped vocabulary with surface form variants (e.g., "walk", "walking", "Walk"), at the expense of less frequent words and multilingual coverage. We show that many of these variations can be captured by transformation vectors -- additive offsets that yield the appropriate word's representation when applied to the base form word embedding -- in both the input and output spaces. Building on this, we propose a compact reshaping of the vocabulary: rather than assigning unique tokens to each surface form, we compose them from shared base form and transformation vectors (e.g., "walked" = "walk" + past tense). We apply our approach to multiple LLMs and across five languages, removing up to 10% of vocabulary entries -- thereby freeing space to allocate new, more diverse tokens. Importantly, we do so while also expanding vocabulary coverage to out-of-vocabulary words, with minimal impact on downstream performance, and without modifying model weights. Our findings motivate a foundational rethinking of vocabulary design, moving from string enumeration to a compositional vocabulary that leverages the underlying structure of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17000v1">Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19504v2">DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Code is available at https://github.com/unites-lab/doge
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD). In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs are accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving the performance of the teacher model, student models distilled from the defensively generated outputs demonstrate catastrophically reduced performance, demonstrating DOGe as a practical safeguard against KD-based model imitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16978v1">Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      We present Lark, a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System (MAS). To address verbosity and stakeholder trade-offs, we integrate four mechanisms: (i) plasticity, which applies concise adjustments to candidate solutions; (ii) duplication and maturation, which copy high-performing candidates and specialize them into new modules; (iii) ranked-choice stakeholder aggregation using influence-weighted Borda scoring; and (iv) compute awareness via token-based penalties that reward brevity. The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores. In a controlled evaluation over 30 rounds comparing 14 systems, Lark Full achieves a mean rank of 2.55 (95% CI [2.17, 2.93]) and a mean composite score of 29.4/50 (95% CI [26.34, 32.46]), finishing Top-3 in 80% of rounds while remaining cost competitive with leading commercial models ($0.016 per task). Paired Wilcoxon tests confirm that all four mechanisms contribute significantly as ablating duplication/maturation yields the largest deficit ({\Delta}Score = 3.5, Cohen's d_z = 2.53, p < 0.001), followed by plasticity ({\Delta}Score = 3.4, d_z = 1.86), ranked-choice voting ({\Delta}Score = 2.4, d_z = 1.20), and token penalties ({\Delta}Score = 2.2, d_z = 1.63). Rather than a formal Markov Decision Process with constrained optimization, Lark is a practical, compute-aware neuroevolutionary loop that scales stakeholder-aligned strategy generation and makes trade-offs transparent through per-step metrics. Our work presents proof-of-concept findings and invites community feedback as we expand toward real-world validation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20548v2">$Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce $Q\sharp$, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized $Q$ function. We propose to learn the optimal $Q$ function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized $Q$-values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, $Q\sharp$ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight $Q\sharp$ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at https://github.com/jinpz/q_sharp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16943v1">Peering Inside the Black Box: Uncovering LLM Errors in Optimization Modelling through Component-Level Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to convert natural language descriptions into mathematical optimization formulations. Current evaluations often treat formulations as a whole, relying on coarse metrics like solution accuracy or runtime, which obscure structural or numerical errors. In this study, we present a comprehensive, component-level evaluation framework for LLM-generated formulations. Beyond the conventional optimality gap, our framework introduces metrics such as precision and recall for decision variables and constraints, constraint and objective root mean squared error (RMSE), and efficiency indicators based on token usage and latency. We evaluate GPT-5, LLaMA 3.1 Instruct, and DeepSeek Math across optimization problems of varying complexity under six prompting strategies. Results show that GPT-5 consistently outperforms other models, with chain-of-thought, self-consistency, and modular prompting proving most effective. Analysis indicates that solver performance depends primarily on high constraint recall and low constraint RMSE, which together ensure structural correctness and solution reliability. Constraint precision and decision variable metrics play secondary roles, while concise outputs enhance computational efficiency. These findings highlight three principles for NLP-to-optimization modeling: (i) Complete constraint coverage prevents violations, (ii) minimizing constraint RMSE ensures solver-level accuracy, and (iii) concise outputs improve computational efficiency. The proposed framework establishes a foundation for fine-grained, diagnostic evaluation of LLMs in optimization modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16933v1">Tutoring LLM into a Better CUDA Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in Euro-Par 2025: Parallel Processing, Part II, and is available online at https://doi.org/10.1007/978-3-031-99857-7_18
    </div>
    <details class="paper-abstract">
      Recent leaps in large language models (LLMs) caused a revolution in programming tools (like GitHub Copilot) that can help with code generation, debugging, and even performance optimization. In this paper, we focus on the capabilities of the most recent reasoning models to generate optimized CUDA code for predefined, well-known tasks. Our objective is to determine which types of code optimizations and parallel patterns the LLMs can perform by themselves and whether they can be improved by tutoring (providing more detailed hints and guidelines in the prompt). The generated solutions were evaluated both automatically (for correctness and speedup) and manually (code reviews) to provide a more detailed perspective. We also tried an interactive approach where the LLM can fix its previous mistakes within a session. The results indicate that LLMs are quite skilled coders; however, they require tutoring to reach optimized solutions provided by parallel computing experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16932v1">Prompt-MII: Meta-Learning Instruction Induction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16916v1">SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer promising capabilities for tackling complex reasoning tasks, including optimization problems. However, existing methods either rely on prompt engineering, which leads to poor generalization across problem types, or require costly supervised training. We introduce SolverLLM, a training-free framework that leverages test-time scaling to solve diverse optimization problems. Rather than solving directly, SolverLLM generates mathematical formulations and translates them into solver-ready code, guided by a novel Monte Carlo Tree Search (MCTS) strategy. To enhance the search process, we modify classical MCTS with (1) dynamic expansion for adaptive formulation generation, (2) prompt backpropagation to guide exploration via outcome-driven feedback, and (3) uncertainty backpropagation to incorporate reward reliability into decision-making. Experiments on six standard benchmark datasets demonstrate that SolverLLM outperforms both prompt-based and learning-based baselines, achieving strong generalization without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16492v1">Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Reliable ML and Regulatable ML workshops, Neurips 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14315v6">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and three additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10906v2">Understanding LLMs' Cross-Lingual Context Retrieval: How Good It Is And Where It Comes From</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Cross-lingual context retrieval (extracting contextual information in one language based on requests in another) is a fundamental aspect of cross-lingual alignment, but the performance and mechanism of it for large language models (LLMs) remains unclear. In this paper, we evaluate the cross-lingual context retrieval of over 40 LLMs across 12 languages, using cross-lingual machine reading comprehension (xMRC) as a representative scenario. Our results show that post-trained open LLMs show strong cross-lingual context retrieval ability, comparable to closed-source LLMs such as GPT-4o, and their estimated oracle performances greatly improve after post-training. Our mechanism analysis shows that the cross-lingual context retrieval process can be divided into two main phases: question encoding and answer retrieval, which are formed in pre-training and post-training respectively. The phasing stability correlates with xMRC performance, and the xMRC bottleneck lies at the last model layers in the second phase, where the effect of post-training can be evidently observed. Our results also indicate that larger-scale pretraining cannot improve the xMRC performance. Instead, larger LLMs need further multilingual post-training to fully unlock their cross-lingual context retrieval potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17598v2">Hallucination Detection in LLMs Using Spectral Features of Attention Maps</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Accepted to EMNLP 2025. Code available at https://github.com/graphml-lab-pwr/lapeigvals
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16418v1">FourierCompress: Layer-Aware Spectral Activation Compression for Efficient and Accurate Collaborative LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Collaborative large language model (LLM) inference enables real-time, privacy-preserving AI services on resource-constrained edge devices by partitioning computational workloads between client devices and edge servers. However, this paradigm is severely hindered by communication bottlenecks caused by the transmission of high-dimensional intermediate activations, exacerbated by the autoregressive decoding structure of LLMs, where bandwidth consumption scales linearly with output length. Existing activation compression methods struggle to simultaneously achieve high compression ratios, low reconstruction error, and computational efficiency. This paper proposes FourierCompress, a novel, layer-aware activation compression framework that exploits the frequency-domain sparsity of LLM activations. We rigorously demonstrate that activations from the first Transformer layer exhibit strong smoothness and energy concentration in the low-frequency domain, making them highly amenable to near-lossless compression via the Fast Fourier Transform (FFT). FourierCompress transforms activations into the frequency domain, retains only a compact block of low-frequency coefficients, and reconstructs the signal at the server using conjugate symmetry, enabling seamless hardware acceleration on DSPs and FPGAs. Extensive experiments on Llama 3 and Qwen2.5 models across 10 commonsense reasoning datasets demonstrate that FourierCompress preserves performance remarkably close to the uncompressed baseline, outperforming Top-k, QR, and SVD. FourierCompress bridges the gap between communication efficiency (an average 7.6x reduction in activation size), near-lossless inference (less than 0.3% average accuracy loss), and significantly faster compression (achieving over 32x reduction in compression time compared to Top-k via hardware acceleration) for edge-device LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16415v1">MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 NeurIPS 2025 poster
    </div>
    <details class="paper-abstract">
      As distributed optimization scales to meet the demands of Large Language Model (LLM) training, hardware failures become increasingly non-negligible. Existing fault-tolerant training methods often introduce significant computational or memory overhead, demanding additional resources. To address this challenge, we propose Memory- and Computation-efficient Fault-tolerant Optimization (MeCeFO), a novel algorithm that ensures robust training with minimal overhead. When a computing node fails, MeCeFO seamlessly transfers its training task to a neighboring node while employing memory- and computation-efficient algorithmic optimizations to minimize the extra workload imposed on the neighboring node handling both tasks. MeCeFO leverages three key algorithmic designs: (i) Skip-connection, which drops the multi-head attention (MHA) module during backpropagation for memory- and computation-efficient approximation; (ii) Recomputation, which reduces activation memory in feedforward networks (FFNs); and (iii) Low-rank gradient approximation, enabling efficient estimation of FFN weight matrix gradients. Theoretically, MeCeFO matches the convergence rate of conventional distributed training, with a rate of $\mathcal{O}(1/\sqrt{nT})$, where n is the data parallelism size and T is the number of iterations. Empirically, MeCeFO maintains robust performance under high failure rates, incurring only a 4.18% drop in throughput, demonstrating 5.0$\times$ to 6.7$\times$ greater resilience than previous SOTA approaches. Codes are available at https://github.com/pkumelon/MeCeFO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v3">How Far Have LLMs Come Toward Automated SATD Taxonomy Construction?</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 5 pages, APSEC 2025
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. In this study, we investigate to what extent large language models (LLMs) can generate SATD taxonomies. We designed a structured, LLM-driven pipeline that mirrors the taxonomy construction steps researchers typically follow. We evaluated it on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovered domain-specific categories reported in prior work, such as Layer Configuration in machine learning. It also completed taxonomy generation in under two hours and for less than $1, even on the largest dataset. These results suggest that, while full automation remains challenging, LLMs can support semi-automated SATD taxonomy construction. Furthermore, our work opens up avenues for future work, such as automated taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12713v2">Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition?</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(https://github.com/Humanity-s-Last-Code-Exam/HLCE).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16395v1">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Development</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong capabilities in software engineering tasks, raising expectations of revolutionary productivity gains. However, enterprise software development is largely driven by incremental evolution, where challenges extend far beyond routine coding and depend critically on tacit knowledge, including design decisions at different levels and historical trade-offs. To achieve effective AI-powered support for complex software development, we should align emerging AI capabilities with the practical realities of enterprise development. To this end, we systematically identify challenges from both software and LLM perspectives. Alongside these challenges, we outline opportunities where AI and structured knowledge frameworks can enhance decision-making in tasks such as issue localization and impact analysis. To address these needs, we propose the Code Digital Twin, a living framework that models both the physical and conceptual layers of software, preserves tacit knowledge, and co-evolves with the codebase. By integrating hybrid knowledge representations, multi-stage extraction pipelines, incremental updates, LLM-empowered applications, and human-in-the-loop feedback, the Code Digital Twin transforms fragmented knowledge into explicit and actionable representations. Our vision positions it as a bridge between AI advancements and enterprise software realities, providing a concrete roadmap toward sustainable, intelligent, and resilient development and evolution of ultra-complex systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16384v1">SemOpt: LLM-Driven Code Optimization via Rule-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Automated code optimization aims to improve performance in programs by refactoring code, and recent studies focus on utilizing LLMs for the optimization. Typical existing approaches mine optimization commits from open-source codebases to construct a large-scale knowledge base, then employ information retrieval techniques such as BM25 to retrieve relevant optimization examples for hotspot code locations, thereby guiding LLMs to optimize these hotspots. However, since semantically equivalent optimizations can manifest in syntactically dissimilar code snippets, current retrieval methods often fail to identify pertinent examples, leading to suboptimal optimization performance. This limitation significantly reduces the effectiveness of existing optimization approaches. To address these limitations, we propose SemOpt, a novel framework that leverages static program analysis to precisely identify optimizable code segments, retrieve the corresponding optimization strategies, and generate the optimized results. SemOpt consists of three key components: (1) A strategy library builder that extracts and clusters optimization strategies from real-world code modifications. (2) A rule generator that generates Semgrep static analysis rules to capture the condition of applying the optimization strategy. (3) An optimizer that utilizes the strategy library to generate optimized code results. All the three components are powered by LLMs. On our benchmark containing 151 optimization tasks, SemOpt demonstrates its effectiveness under different LLMs by increasing the number of successful optimizations by 1.38 to 28 times compared to the baseline. Moreover, on popular large-scale C/C++ projects, it can improve individual performance metrics by 5.04% to 218.07%, demonstrating its practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20999v4">MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. Typically, LLMs are first pre-trained on large corpora and subsequently fine-tuned on task-specific datasets. However, during fine-tuning, LLMs may forget some knowledge acquired in the pre-training stage, leading to a decline in general capabilities. Existing approaches to mitigate forgetting often rely on access to pre-training data, which may be unavailable in many real-world scenarios--such as fine-tuning checkpoint-only open-source LLMs. To address this challenge, we propose a new fine-tuning algorithm termed Momentum-Filtered Optimizer (MoFO). MoFO is an extension of greedy block coordinate descent (BCD) methods: in each iteration, MoFO only updates the model parameters with the largest momentum magnitudes, while keeping all other parameters fixed. MoFO achieves similar fine-tuning performance to the default fine-tuning algorithm while effectively mitigating knowledge forgetting. We validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its effectiveness in mitigating forgetting without pre-training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13586v2">Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has opened new opportunities for cre- ating dynamic non-player characters (NPCs) in gaming environments, enabling both func- tional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which eval- uates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16373v1">Navigating through the hidden embedding space: steering LLMs to improve mental health assessment</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Models (LLMs) is transforming AI, opening new opportunities in sensitive and high-impact areas such as Mental Health (MH). Yet, despite these advancements, recent evidence reveals that smaller-scale models still struggle to deliver optimal performance in domain-specific applications. In this study, we present a cost-efficient yet powerful approach to improve MH assessment capabilities of an LLM, without relying on any computationally intensive techniques. Our lightweight method consists of a linear transformation applied to a specific layer's activations, leveraging steering vectors to guide the model's output. Remarkably, this intervention enables the model to achieve improved results across two distinct tasks: (1) identifying whether a Reddit post is useful for detecting the presence or absence of depressive symptoms (relevance prediction task), and (2) completing a standardized psychological screening questionnaire for depression based on users' Reddit post history (questionnaire completion task). Results highlight the untapped potential of steering mechanisms as computationally efficient tools for LLMs' MH domain adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16366v1">Integrating LLM and Diffusion-Based Agents for Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 10 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Agent-based social simulation provides a valuable methodology for predicting social information diffusion, yet existing approaches face two primary limitations. Traditional agent models often rely on rigid behavioral rules and lack semantic understanding of textual content, while emerging large language model (LLM)-based agents incur prohibitive computational costs at scale. To address these challenges, we propose a hybrid simulation framework that strategically integrates LLM-driven agents with diffusion model-based agents. The framework employs LLM-based agents to simulate a core subset of users with rich semantic reasoning, while a diffusion model handles the remaining population efficiently. Although the two agent types operate on disjoint user groups, both incorporate key factors including user personalization, social influence, and content awareness, and interact through a coordinated simulation process. Extensive experiments on three real-world datasets demonstrate that our framework outperforms existing methods in prediction accuracy, validating the effectiveness of its modular design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06807v2">MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Private large language model (LLM) inference based on secure multi-party computation (MPC) achieves formal data privacy protection but suffers from significant latency overhead, especially for long input sequences. While key-value (KV) cache eviction and sparse attention algorithms have been proposed for efficient LLM inference in plaintext, they are not designed for MPC and cannot benefit private LLM inference directly. In this paper, we propose an accurate and MPC-friendly KV cache eviction framework, dubbed MPCache, building on the observation that historical tokens in a long sequence may have different effects on the downstream decoding. Hence, MPCache combines a look-once static eviction algorithm to discard unimportant KV cache and a query-aware dynamic selection algorithm to activate only a small subset of KV cache for attention computation. MPCache further incorporates a series of optimizations for efficient dynamic KV cache selection, including MPC-friendly similarity approximation, hierarchical KV cache clustering, and cross-layer index-sharing strategy. Extensive experiments demonstrate that MPCache consistently outperforms prior-art KV cache eviction baselines across different generation tasks and achieves 1.8 ~ 2.01x and 3.39 ~ 8.37x decoding latency and communication reduction on different sequence lengths, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25835v3">Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Under Review; Add codebase
    </div>
    <details class="paper-abstract">
      Test-time scaling improves large language models (LLMs) on long-horizon reasoning tasks by allocating more compute at inference. LLM Inference via Tree Search (LITS) methods achieve strong performance but are highly inefficient, often running an order of magnitude slower than iterative approaches. We propose Chain-in-Tree (CiT), a plug-in framework that decides when to branch during search rather than expanding at every step. CiT introduces lightweight Branching Necessity (BN) evaluations: BN-DP (Direct Prompting), where an auxiliary LLM judges branching needs, and BN-SC (Self-Consistency), which clusters candidate actions to assess agreement. Integrated into Tree of Thoughts, ReST-MCTS, and RAP, BN-DP achieves 75-85% reductions in token generation, model calls, and runtime on GSM8K and Math500, with often negligible or no accuracy loss. BN-SC typically yields substantial savings (up to 80%) generally but shows instability in 1-4 out of 14 settings, caused by a small subset of examples that produce extremely long reasoning steps. We theoretically prove that BN-DP never increases policy invocations and release both modular LITS implementations and a lightweight CiT function applicable across all LITS variants. The full codebase is publicly available at https://github.com/xinzhel/chain_in_tree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07230v2">Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v4">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. Our code is available at https://github.com/IANNXANG/RuscaRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v3">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17333v3">Whose Journey Matters? Investigating Identity Biases in Large Language Models (LLMs) for Travel Planning Assistance</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integral to the hospitality and tourism industry, concerns about their fairness in serving diverse identity groups persist. Grounded in social identity theory and sociotechnical systems theory, this study examines ethnic and gender biases in travel recommendations generated by LLMs. Using fairness probing, we analyze outputs from three leading open-source LLMs. The results show that test accuracy for both ethnicity and gender classifiers exceed random chance. Analysis of the most influential features reveals the presence of stereotype bias in LLM-generated recommendations. We also found hallucinations among these features, occurring more frequently in recommendations for minority groups. These findings indicate that LLMs exhibit ethnic and gender bias when functioning as travel planning assistants. This study underscores the need for bias mitigation strategies to improve the inclusivity and reliability of generative AI-driven travel planning assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v5">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21184v2">BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated (and then estimated) in a principled way using a probabilistic model derived from the LLM's predictive distributions and provide detailed insights into key decisions in its construction and updating procedure. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20 questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13776v2">A Knapsack by Any Other Name: Presentation impacts LLM performance on NP-hard problems</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 24 pages, 6 figures, EMNLP 2025
    </div>
    <details class="paper-abstract">
      To investigate the effect of problem presentation on LLMs' ability to solve optimization problems, we introduce the dataset of Everyday Hard Optimization Problems (EHOP), a collection of NP-hard problems expressed in natural language. EHOP includes problem formulations that could be found in computer science textbooks (e.g., graph coloring), versions that are dressed up as problems that could arise in real life (e.g., party planning), and variants with inverted rules. We find that state-of-the-art LLMs, across multiple prompting strategies, systematically solve textbook problems more accurately than their real-life and inverted counterparts. While reasoning models are more capable, they nonetheless show high variance across problem presentations, suggesting they lack a truly robust reasoning mechanism. We argue that this constitutes evidence that LLMs are still heavily dependent on what was seen in training and struggle to generalize to novel problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16645v1">Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16614v1">Count Counts: Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become a compelling way to strengthen the multi step reasoning ability of Large Language Models (LLMs). However, prevalent RL paradigms still lean on sparse outcome-based rewards and limited exploration, which often drives LLMs toward repetitive and suboptimal reasoning patterns. In this paper, we study the central question of how to design exploration for LLM reasoning and introduce MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards), a novel RL algorithm that augments policy optimization with a principled intrinsic reward. Building on the idea of count-based exploration, MERCI leverages a lightweight Coin Flipping Network (CFN) to estimate the pseudo count and further epistemic uncertainty over reasoning trajectories, and converts them into an intrinsic reward that values novelty while preserving the learning signal from task rewards. We integrate MERCI into some advanced RL frameworks like Group Relative Policy Optimization (GRPO). Experiments on complex reasoning benchmarks demonstrate that MERCI encourages richer and more varied chains of thought, significantly improves performance over strong baselines, and helps the policy escape local routines to discover better solutions. It indicates that our targeted intrinsic motivation can make exploration reliable for language model reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16610v1">Structuring Security: A Survey of Cybersecurity Ontologies, Semantic Log Processing, and LLMs Application</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      This survey investigates how ontologies, semantic log processing, and Large Language Models (LLMs) enhance cybersecurity. Ontologies structure domain knowledge, enabling interoperability, data integration, and advanced threat analysis. Security logs, though critical, are often unstructured and complex. To address this, automated construction of Knowledge Graphs (KGs) from raw logs is emerging as a key strategy for organizing and reasoning over security data. LLMs enrich this process by providing contextual understanding and extracting insights from unstructured content. This work aligns with European Union (EU) efforts such as NIS 2 and the Cybersecurity Taxonomy, highlighting challenges and opportunities in intelligent ontology-driven cyber defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13982v2">Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11218v2">The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Code: https://github.com/WorldHellow/SLAQ/tree/main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09986v3">From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Motivated by the remarkable success of artificial intelligence (AI) across diverse fields, the application of AI to solve scientific problems, often formulated as partial differential equations (PDEs), has garnered increasing attention. While most existing research concentrates on theoretical properties (such as well-posedness, regularity, and continuity) of the solutions, alongside direct AI-driven methods for solving PDEs, the challenge of uncovering symbolic relationships within these equations remains largely unexplored. In this paper, we propose leveraging large language models (LLMs) to learn such symbolic relationships. Our results demonstrate that LLMs can effectively predict the operators involved in PDE solutions by utilizing the symbolic information in the PDEs both theoretically and numerically. Furthermore, we show that discovering these symbolic relationships can substantially improve both the efficiency and accuracy of symbolic machine learning for finding analytical approximation of PDE solutions, delivering a fully interpretable solution pipeline. This work opens new avenues for understanding the symbolic structure of scientific problems and advancing their solution processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16590v1">Atom-anchored LLMs speak Chemistry: A Retrosynthesis Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Alan Kai Hassen and Andrius Bernatavicius contributed equally to this work
    </div>
    <details class="paper-abstract">
      Applications of machine learning in chemistry are often limited by the scarcity and expense of labeled data, restricting traditional supervised methods. In this work, we introduce a framework for molecular reasoning using general-purpose Large Language Models (LLMs) that operates without requiring labeled training data. Our method anchors chain-of-thought reasoning to the molecular structure by using unique atomic identifiers. First, the LLM performs a one-shot task to identify relevant fragments and their associated chemical labels or transformation classes. In an optional second step, this position-aware information is used in a few-shot task with provided class examples to predict the chemical transformation. We apply our framework to single-step retrosynthesis, a task where LLMs have previously underperformed. Across academic benchmarks and expert-validated drug discovery molecules, our work enables LLMs to achieve high success rates in identifying chemically plausible reaction sites ($\geq90\%$), named reaction classes ($\geq40\%$), and final reactants ($\geq74\%$). Beyond solving complex chemical tasks, our work also provides a method to generate theoretically grounded synthetic datasets by mapping chemical knowledge onto the molecular structure and thereby addressing data scarcity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v4">Unlocking LLM Repair Capabilities Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06767v2">Beyond Surface Similarity: Evaluating LLM-Based Test Refactorings with Structural and Semantic Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to refactor unit tests, improving readability and structure while preserving behavior. Evaluating such refactorings, however, remains difficult: metrics like CodeBLEU penalize beneficial renamings and edits, while semantic similarities overlook readability and modularity. We propose CTSES, a first step toward human-aligned evaluation of refactored tests. CTSES combines CodeBLEU, METEOR, and ROUGE-L into a composite score that balances semantics, lexical clarity, and structural alignment. Evaluated on 5,000+ refactorings from Defects4J and SF110 (GPT-4o and Mistral-Large), CTSES reduces false negatives and provides more interpretable signals than individual metrics. Our emerging results illustrate that CTSES offers a proof-of-concept for composite approaches, showing their promise in bridging automated metrics and developer judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22961v2">ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16559v1">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16552v1">LANPO: Bootstrapping Language and Numerical Feedback for Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning in large language models (LLMs) often relies on scalar rewards, a practice that discards valuable textual rationale buried in the rollouts, forcing the model to explore \textit{de novo} with each attempt and hindering sample efficiency. While LLMs can uniquely learn from language feedback provided in-context, naively integrating on-line experiences into RL training presents a paradox: feedback from the same problem risks information leakage and memorization, while feedback from different problems often leads to behavior collapse due to irrelevant context. To resolve this tension, we propose \textbf{Language-And-Numerical Policy Optimization (LANPO)}, a framework that cleanly separates the roles of feedback: language guides exploration, while numerical rewards drive optimization. LANPO builds a dynamic experience pool from past trials and introduces two principles to ensure feedback is effective: \emph{Reward-Agnostic Reflection} for safe intra-sample self-correction and \emph{Relevant Abstraction} to distill generalizable lessons from inter-sample experiences. Across mathematical reasoning benchmarks, LANPO enables 7B and 14B models to significantly outperform strong baselines trained with GRPO in test accuracy. Our work provides a robust method for integrating historical experiences into the LLM RL loop, creating more effective and data-efficient learning agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16551v1">From Reviews to Actionable Insights: An LLM-Based Approach for Attribute and Feature Extraction</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      This research proposes a systematic, large language model (LLM) approach for extracting product and service attributes, features, and associated sentiments from customer reviews. Grounded in marketing theory, the framework distinguishes perceptual attributes from actionable features, producing interpretable and managerially actionable insights. We apply the methodology to 20,000 Yelp reviews of Starbucks stores and evaluate eight prompt variants on a random subset of reviews. Model performance is assessed through agreement with human annotations and predictive validity for customer ratings. Results show high consistency between LLMs and human coders and strong predictive validity, confirming the reliability of the approach. Human coders required a median of six minutes per review, whereas the LLM processed each in two seconds, delivering comparable insights at a scale unattainable through manual coding. Managerially, the analysis identifies attributes and features that most strongly influence customer satisfaction and their associated sentiments, enabling firms to pinpoint "joy points," address "pain points," and design targeted interventions. We demonstrate how structured review data can power an actionable marketing dashboard that tracks sentiment over time and across stores, benchmarks performance, and highlights high-leverage features for improvement. Simulations indicate that enhancing sentiment for key service features could yield 1-2% average revenue gains per store.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16549v1">ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. Recent work has focused on using LLMs to improve review efficiency or generate insightful review content. However, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine the peer review ecosystem and compromise academic integrity. To address this critical issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews. ReviewGuard employs a comprehensive four-stage LLM-driven framework that: (1) collects ICLR and NeurIPS papers with their corresponding reviews from OpenReview; (2) annotates review types using GPT-4.1 with human validation; (3) addresses class imbalance and data scarcity through LLM-driven synthetic data augmentation, producing a final corpus of 6,634 papers, 24,657 real reviews, and 46,438 synthetic reviews; and (4) fine-tunes both encoder-based models and open source LLMs. We perform comprehensive feature analysis of the structure and quality of the review text. Compared to sufficient reviews, deficient reviews demonstrate lower rating scores, higher self-reported confidence, reduced structural complexity, and a higher proportion of negative sentiment. AI-generated text detection reveals that, since ChatGPT's emergence, AI-generated reviews have increased dramatically. In the evaluation of deficient review detection models, mixed training with synthetic and real review data provides substantial enhancements to recall and F1 scores on the binary task. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review while offering valuable insights into human-AI collaboration to maintain academic integrity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16530v1">Realizing LLMs' Causal Potential Requires Science-Grounded, Novel Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent claims of strong performance by Large Language Models (LLMs) on causal discovery are undermined by a key flaw: many evaluations rely on benchmarks likely included in pretraining corpora. Thus, apparent success suggests that LLM-only methods, which ignore observational data, outperform classical statistical approaches. We challenge this narrative by asking: Do LLMs truly reason about causal structure, and how can we measure it without memorization concerns? Can they be trusted for real-world scientific discovery? We argue that realizing LLMs' potential for causal analysis requires two shifts: (P.1) developing robust evaluation protocols based on recent scientific studies to guard against dataset leakage, and (P.2) designing hybrid methods that combine LLM-derived knowledge with data-driven statistics. To address P.1, we encourage evaluating discovery methods on novel, real-world scientific studies. We outline a practical recipe for extracting causal graphs from recent publications released after an LLM's training cutoff, ensuring relevance and preventing memorization while capturing both established and novel relations. Compared to benchmarks like BNLearn, where LLMs achieve near-perfect accuracy, they perform far worse on our curated graphs, underscoring the need for statistical grounding. Supporting P.2, we show that using LLM predictions as priors for the classical PC algorithm significantly improves accuracy over both LLM-only and purely statistical methods. We call on the community to adopt science-grounded, leakage-resistant benchmarks and invest in hybrid causal discovery methods suited to real-world inquiry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05909v2">Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Open-source code available at https://github.com/flowersteam/llm_persuasion
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16967v2">Hard Negatives, Hard Lessons: Revisiting Training Data Quality for Robust Information Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection, reduces the training set size by 2.35$\times$, surprisingly increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We utilize LLMs as a simple, cost-effective approach to identify and relabel false negatives in training datasets. Experimental results show that relabeling false negatives as true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7$\unicode{x2013}$1.4 points on BEIR and by 1.7$\unicode{x2013}$1.8 points at nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of LLMs to identify false negatives is supported by human annotation results. Our training dataset and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17891v1">TritonRL: Training LLMs to Think and Code Triton Without Cheating</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      With the rapid evolution of large language models (LLMs), the demand for automated, high-performance system kernels has emerged as a key enabler for accelerating development and deployment. We introduce TritonRL, a domain-specialized LLM for Triton kernel generation, trained with a novel training framework that enables robust and automated kernel synthesis. Unlike general-purpose programming languages, Triton kernel generation faces unique challenges due to data scarcity and incomplete evaluation criteria, vulnerable to reward hacking. Our approach addresses these challenges end-to-end by distilling Triton-specific knowledge through supervised fine-tuning on curated datasets, and further improving code quality via reinforcement learning (RL) with robust, verifiable rewards and hierarchical reward assignment. Our RL framework robustly detects reward hacking and guides both reasoning traces and code tokens through fine-grained verification and hierarchical reward decomposition, enabling the model to generate high-quality Triton kernels that can truly replace existing modules. With robust and fine-grained evaluation, our experiments on KernelBench demonstrate that TritonRL achieves state-of-the-art correctness and speedup, surpassing all other Triton-specific models and underscoring the effectiveness of our RL-based training paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15428v1">Fault Cause Identification across Manufacturing Lines through Ontology-Guided and Process-Aware FMEA Graph Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Fault cause identification in automated manufacturing lines is challenging due to the system's complexity, frequent reconfigurations, and the limited reusability of existing Failure Mode and Effects Analysis (FMEA) knowledge. Although FMEA worksheets contain valuable expert insights, their reuse across heterogeneous lines is hindered by natural language variability, inconsistent terminology, and process differences. To address these limitations, this study proposes a process-aware framework that enhances FMEA reusability by combining manufacturing-domain conceptualization with graph neural network (GNN) reasoning. First, FMEA worksheets from multiple manufacturing lines are transformed into a unified knowledge graph through ontology-guided large language model (LLM) extraction, capturing domain concepts such as actions, states, components, and parameters. Second, a Relational Graph Convolutional Network (RGCN) with the process-aware scoring function learns embeddings that respect both semantic relationships and sequential process flows. Finally, link prediction is employed to infer and rank candidate fault causes consistent with the target line's process flow. A case study on automotive pressure sensor assembly lines demonstrates that the proposed method outperforms a state-of-the-art retrieval-augmented generation (RAG) baseline (F1@20 = 0.267) and an RGCN approach (0.400), achieving the best performance (0.523) in fault cause identification. Ablation studies confirm the contributions of both LLM-driven domain conceptualization and process-aware learning. These results indicate that the proposed framework significantly improves the transferability of FMEA knowledge across heterogeneous lines, thereby supporting operators in diagnosing failures more reliably and paving the way for future domain-adaptive LLM applications in smart manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15414v1">MARS: Reinforcing Multi-Agent Reasoning of LLMs through Self-Play in Strategic Games</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Developing Large Language Models (LLMs) to cooperate and compete effectively within multi-agent systems is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARS, an end-to-end RL framework that incentivizes Multi-Agent Reasoning of LLMs through Self-play in both cooperative and competitive games. MARS features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, the MARS agent trained from Qwen3-4B develops strong strategic abilities that generalize to held-out games with up to 28.7% performance improvements. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of multi-agent systems in reasoning benchmarks. When integrated into leading multi-agent systems, our MARS agent achieves significant performance gains of 10.0% on AIME and 12.5% on GPQA-Diamond. These results establish end-to-end RL training with self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs. Our code and models are publicly available at https://github.com/thu-nics/MARS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26490v2">VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 The code, dataset, and leaderboard are available at https://vitabench.github.io/
    </div>
    <details class="paper-abstract">
      As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at https://vitabench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15406v1">VocalBench-DF: A Benchmark for Evaluating Speech LLM Robustness to Disfluency</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While Speech Large Language Models (Speech-LLMs) show strong performance in many applications, their robustness is critically under-tested, especially to speech disfluency. Existing evaluations often rely on idealized inputs, overlooking common disfluencies, particularly those associated with conditions like Parkinson's disease. This work investigates whether current Speech-LLMs can maintain performance when interacting with users who have speech impairments. To facilitate this inquiry, we introduce VocalBench-DF, a framework for the systematic evaluation of disfluency across a multi-dimensional taxonomy. Our evaluation of 22 mainstream Speech-LLMs reveals substantial performance degradation, indicating that their real-world readiness is limited. Further analysis identifies phoneme-level processing and long-context modeling as primary bottlenecks responsible for these failures. Strengthening recognition and reasoning capability from components and pipelines can substantially improve robustness. These findings highlight the urgent need for new methods to improve disfluency handling and build truly inclusive Speech-LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18492v2">GuardReasoner: Towards Reasoning-based LLM Safeguards</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 22 pages, 18 figures
    </div>
    <details class="paper-abstract">
      As LLMs increasingly impact safety-critical applications, ensuring their safety using guardrails remains a key challenge. This paper proposes GuardReasoner, a new safeguard for LLMs, by guiding the guard model to learn to reason. Concretely, we first create the GuardReasonerTrain dataset, which consists of 127K samples with 460K detailed reasoning steps. Then, we introduce reasoning SFT to unlock the reasoning capability of guard models. In addition, we present hard sample DPO to further strengthen their reasoning ability. In this manner, GuardReasoner achieves better performance, explainability, and generalizability. Extensive experiments and analyses on 13 benchmarks of 3 guardrail tasks demonstrate its superiority. Remarkably, GuardReasoner 8B surpasses GPT-4o+CoT by 5.74% and LLaMA Guard 3 8B by 20.84% F1 score on average. We release the training data, code, and models with different scales (1B, 3B, 8B) of GuardReasoner : https://github.com/yueliu1999/GuardReasoner/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01077v2">Zero-Shot Document-Level Biomedical Relation Extraction via Scenario-based Prompt Design in Two-Stage with LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      With the advent of artificial intelligence (AI), many researchers are attempting to extract structured information from document-level biomedical literature by fine-tuning large language models (LLMs). However, they face significant challenges such as the need for expensive hardware, like high-performance GPUs and the high labor costs associated with annotating training datasets, especially in biomedical realm. Recent research on LLMs, such as GPT-4 and Llama3, has shown promising performance in zero-shot settings, inspiring us to explore a novel approach to achieve the same results from unannotated full documents using general LLMs with lower hardware and labor costs. Our approach combines two major stages: named entity recognition (NER) and relation extraction (RE). NER identifies chemical, disease and gene entities from the document with synonym and hypernym extraction using an LLM with a crafted prompt. RE extracts relations between entities based on predefined relation schemas and prompts. To enhance the effectiveness of prompt, we propose a five-part template structure and a scenario-based prompt design principles, along with evaluation method to systematically assess the prompts. Finally, we evaluated our approach against fine-tuning and pre-trained models on two biomedical datasets: ChemDisGene and CDR. The experimental results indicate that our proposed method can achieve comparable accuracy levels to fine-tuning and pre-trained models but with reduced human and hardware expenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03345v2">Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted for publication in the Journal of Systems and Software (Special Issue on Reliable and Secure Large Language Models for Software Engineering)
    </div>
    <details class="paper-abstract">
      The swift spread of fake news and disinformation campaigns poses a significant threat to public trust, political stability, and cybersecurity. Traditional Cyber Threat Intelligence (CTI) approaches, which rely on low-level indicators such as domain names and social media handles, are easily evaded by adversaries who frequently modify their online infrastructure. To address these limitations, we introduce a novel CTI framework that focuses on high-level, semantic indicators derived from recurrent narratives and relationships of disinformation campaigns. Our approach extracts structured CTI indicators from unstructured disinformation content, capturing key entities and their contextual dependencies within fake news using Large Language Models (LLMs). We further introduce FakeCTI, the first dataset that systematically links fake news to disinformation campaigns and threat actors. To evaluate the effectiveness of our CTI framework, we analyze multiple fake news attribution techniques, spanning from traditional Natural Language Processing (NLP) to fine-tuned LLMs. This work shifts the focus from low-level artifacts to persistent conceptual structures, establishing a scalable and adaptive approach to tracking and countering disinformation campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05846v2">EMCee: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 under review, 21pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive progress across a wide range of tasks, yet their heavy reliance on English-centric training data leads to significant performance degradation in non-English languages. While existing multilingual prompting methods emphasize reformulating queries into English or enhancing reasoning capabilities, they often fail to incorporate the language- and culture-specific grounding that is essential for some queries. To address this limitation, we propose EMCee (Extracting synthetic Multilingual Context and merging), a simple yet effective framework that enhances the multilingual capabilities of LLMs by explicitly extracting and utilizing query-relevant knowledge from the LLM itself. In particular, EMCee first extracts synthetic context to uncover latent, language-specific knowledge encoded within the LLM, and then dynamically merges this contextual insight with reasoning-oriented outputs through a judgment-based selection mechanism. Extensive experiments on four multilingual benchmarks covering diverse languages and tasks demonstrate that EMCee consistently outperforms prior approaches, achieving an average relative improvement of 16.4% overall and 31.7% in low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00615v2">ACON: Optimizing Context Compression for Long-horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. Our code is available at https://github.com/microsoft/acon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15346v1">When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Ensembling Large Language Models (LLMs) has gained attention as a promising approach to surpass the performance of individual models by leveraging their complementary strengths. In particular, aggregating models' next-token probability distributions to select the next token has been shown to be effective in various tasks. However, while successful for short-form answers, its application to long-form generation remains underexplored. In this paper, we show that using existing ensemble methods in long-form generation requires a careful choice of ensembling positions, since the standard practice of ensembling at every token often degrades performance. We identify two key factors for determining these positions: tokenization mismatch across models and consensus in their next-token probability distributions. Based on this, we propose SAFE, (Stable And Fast LLM Ensembling), a framework that selectively ensembles by jointly considering these factors. To further improve stability, we introduce a probability sharpening strategy that consolidates probabilities spread across multiple sub-word tokens representing the same word into a single representative token. Our experiments on diverse benchmarks, including MATH500 and BBH, demonstrate that SAFE outperforms existing methods in both accuracy and efficiency, with gains achieved even when ensembling fewer than 1% of tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15330v1">BeLLMan: Controlling LLM Congestion</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 To be presented at FAISYS 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) applications are blindfolded to the infrastructure underneath and generate tokens autoregressively, indifferent to the system load, thus risking inferencing latency inflation and poor user experience. Our first-cut controller, named beLLMan, enables the LLM infrastructure to actively and progressively signal the first-party LLM application to adjust the output length in response to changing system load. On a real testbed with H100 GPUs, beLLMan helps keep inferencing latency under control (upto 8X lower end-to-end latency) and reduces energy consumption by 25% (while serving 19% more requests) during periods of congestion for a summarization workload.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15306v1">WebGen-V Bench: Structured Representation for Enhancing Visual Design in LLM-based Web Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Witnessed by the recent advancements on leveraging LLM for coding and multimodal understanding, we present WebGen-V, a new benchmark and framework for instruction-to-HTML generation that enhances both data quality and evaluation granularity. WebGen-V contributes three key innovations: (1) an unbounded and extensible agentic crawling framework that continuously collects real-world webpages and can leveraged to augment existing benchmarks; (2) a structured, section-wise data representation that integrates metadata, localized UI screenshots, and JSON-formatted text and image assets, explicit alignment between content, layout, and visual components for detailed multimodal supervision; and (3) a section-level multimodal evaluation protocol aligning text, layout, and visuals for high-granularity assessment. Experiments with state-of-the-art LLMs and ablation studies validate the effectiveness of our structured data and section-wise evaluation, as well as the contribution of each component. To the best of our knowledge, WebGen-V is the first work to enable high-granularity agentic crawling and evaluation for instruction-to-HTML generation, providing a unified pipeline from real-world data acquisition and webpage generation to structured multimodal assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15283v1">Exemplar-Guided Planing: Enhanced LLM Agent for KGQA</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as interactive agents show significant promise in Knowledge Graph Question Answering (KGQA) but often struggle with the semantic gap between natural language queries and structured knowledge graph (KG) representations. This leads to suboptimal planning and inefficient exploration on KG, while training-free approaches often underutilize valuable reasoning patterns in training data. To address these limitations, we propose a novel framework, Exemplar-Guided Planning (EGP), which enhances the planning capabilities of LLM agents for KGQA. EGP first preprocesses the training set questions via entity templating to normalize semantic variations. It then retrieves highly similar exemplary questions and their successful reasoning paths from this preprocessed set using semantic embeddings and an efficient FAISS index. These retrieved exemplars dynamically guide the LLM's planning process in two key phases: (1) Task Decomposition, by aligning generated sub-objectives with proven reasoning steps, and (2) Relation Exploration, by providing high-quality auxiliary information to improve relation pruning accuracy. Additionally, we introduce a Smart Lookahead mechanism during relation exploration to improve efficiency by preemptively exploring promising paths and potentially terminating exploration earlier. We apply EGP to the Plan-on-Graph (PoG) framework, termed PoG-EGP. Extensive experiments on two real-world KGQA datasets, WebQSP and CWQ, demonstrate that PoG-EGP significantly improves over the baseline PoG system and other compared methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14005v2">PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 The code is available at https://github.com/weizou52/PIShield
    </div>
    <details class="paper-abstract">
      LLM-integrated applications are vulnerable to prompt injection attacks, where an attacker contaminates the input to inject malicious prompts, causing the LLM to follow the attacker's intent instead of the original user's. Existing prompt injection detection methods often have sub-optimal performance and/or high computational overhead. In this work, we propose PIShield, a detection method that is both effective and efficient. Our key observation is that the internal representation of the final token in a prompt-extracted from a specific layer of the LLM, which we term the injection-critical layer-captures distinguishing features between clean and contaminated prompts. Leveraging this insight, we train a simple linear classifier on these internal representations using a labeled set of clean and contaminated prompts. We compare PIShield against 11 baselines across 5 diverse benchmark datasets and 8 prompt injection attacks. The results demonstrate that PIShield is both highly effective and efficient, substantially outperforming existing methods. Additionally, we show that PIShield resists strong adaptive attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15261v1">AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 LAW 2025 Workshop at NeurIPS 2025. Work done from late 2023 to early 2024
    </div>
    <details class="paper-abstract">
      Riding on the success of LLMs with retrieval-augmented generation (RAG), there has been a growing interest in augmenting agent systems with external memory databases. However, the existing systems focus on storing text information in their memory, ignoring the importance of multimodal signals. Motivated by the multimodal nature of human memory, we present AUGUSTUS, a multimodal agent system aligned with the ideas of human memory in cognitive science. Technically, our system consists of 4 stages connected in a loop: (i) encode: understanding the inputs; (ii) store in memory: saving important information; (iii) retrieve: searching for relevant context from memory; and (iv) act: perform the task. Unlike existing systems that use vector databases, we propose conceptualizing information into semantic tags and associating the tags with their context to store them in a graph-structured multimodal contextual memory for efficient concept-driven retrieval. Our system outperforms the traditional multimodal RAG approach while being 3.5 times faster for ImageNet classification and outperforming MemGPT on the MSC benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15258v1">Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 14 pages, 7 figures, 40 references
    </div>
    <details class="paper-abstract">
      In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15216v1">Soundness-Aware Level: A Microscopic Signature that Predicts LLM Reasoning Potential</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) can elicit strong reasoning in large language models (LLMs), while their performance after RLVR varies dramatically across different base models. This raises a fundamental question: what microscopic property of pre-trained models leads to this variation? To investigate, we formalize reasoning as chains of Horn clauses ("if-then" rules) built from features extracted from the LLM's latent space via cross-layer sparse autoencoders (SAEs). We estimate the transition probabilities between its features, and further categorize each rule by its semantic soundness level (e.g., strict, plausible, noisy) with an LLM. Our key discovery is that high-potential models are inherently soundness-aware: their internal probability distributions systematically shift across rules' soundness levels, becoming highly distinct for "strict" versus "noisy" rules. In contrast, weaker models are soundness-agnostic, collapsing to one distribution regardless of soundness levels. To quantify this, we introduce the Soundness-Aware Level (SAL), a microscopic metric using the Jensen-Shannon Divergence to measure the separation between these distributions. We show that SAL's predictions of post-RLVR reasoning performance follow a precise empirical law (R^2=0.87) across diverse model families (Qwen, Mistral, Llama, DeepSeek) and scales (0.5B-14B). This reveals that a model's reasoning potential is tied to its intrinsic, pre-trained ability to distinguish sound knowledge from unsound ones. These findings underscore the critical role of model pre-training in shaping reasoning and offer a practical metric grounded in the model's internal mechanisms for selecting/designing stronger base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09585v3">Elevating Visual Perception in Multimodal LLMs with Visual Embedding Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project Page: https://praeclarumjj3.github.io/visper_lm/
    </div>
    <details class="paper-abstract">
      In recent times, the standard practice for developing MLLMs is to feed features from vision encoder(s) into the LLM and train with natural language supervision. This approach often causes models to lean towards language comprehension and undermine the rich visual perception signals present in the data, which are critical for tasks involving spatial reasoning in the domain of embodied AI and robotics. Is it possible to optimize both at the same time? In this work, we propose VisPer-LM, the first approach that infuses visual perception knowledge from expert vision encoders into the LLM's (of an MLLM) hidden representations. We start by investigating MLLMs trained solely with natural language supervision and identify a positive correlation between the quality of visual representations within these models and their downstream performance. Given this insight, we formulate the objective during the pretraining stage in MLLMs as a coupled optimization of predictive visual embedding and next (text) token prediction. Moreover, through extensive probing, we observe improved visual representation quality due to embedding optimization, underscoring the effectiveness of our probing setup. We demonstrate that our VisPer-LM outperforms the single and multi-encoder baselines, proving our approach's superiority over explicitly feeding the corresponding features to the LLM. In particular, VisPer-LM boosts performance by an average margin of up to 2.5% on various benchmarks, with a notable improvement of 8.7% on the Depth task in CV-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12535v2">CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 42 pages, 9 tables
    </div>
    <details class="paper-abstract">
      Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby reducing spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma-2 2B and LLaMA-3.1 8B, notably achieving a +3.3% improvement in MMLU performance with 4000 samples and a +27.2% improvement in HarmBench with only 108 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlation-based selection as an effective and scalable approach for automated SAE steering across language model applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17792v3">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/sftekin/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12997v2">Max It or Miss It: Benchmarking LLM On Solving Extremal Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Our benchmark dataset is available at https://huggingface.co/datasets/binxingao/extrem-bench
    </div>
    <details class="paper-abstract">
      Test-time scaling has enabled Large Language Models (LLMs) with remarkable reasoning capabilities, particularly in mathematical domains, through intermediate chain-of-thought (CoT) reasoning before generating final answers. However, the specific sources and mechanisms underlying these reasoning capabilities remain insufficiently understood. Optimization reasoning, i.e. finding extrema under constraints, represents a fundamental abstraction that underpins critical applications in planning, control, resource allocation, and prompt search. To systematically evaluate this capability, we introduce ExtremBench, a benchmark dataset for solving mathematical extremal problems, curated from inequality exercises used for Chinese Mathematical Olympiad and transformed into $93$ standardized extrema-finding problems. We conduct extensive evaluations across various state-of-the-art open-source model families, including the Qwen3, GPT-OSS, and DeepSeek. Our results reveal that LLMs' extremal-solving reasoning capabilities do not always align with those of current mathematical benchmarks such as AIME25 and MATH-500, with some models showing strong general mathematical reasoning but poor extremal-solving skills, and vice versa. This discrepancy highlights a critical gap in current evaluation practices and suggests that existing benchmarks may not comprehensively capture the full spectrum of mathematical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10343v2">Code Execution as Grounded Supervision for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19131v2">ZeST: an LLM-based Zero-Shot Traversability Navigation for Unknown Environments</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      The advancement of robotics and autonomous navigation systems hinges on the ability to accurately predict terrain traversability. Traditional methods for generating datasets to train these prediction models often involve putting robots into potentially hazardous environments, posing risks to equipment and safety. To solve this problem, we present ZeST, a novel approach leveraging visual reasoning capabilities of Large Language Models (LLMs) to create a traversability map in real-time without exposing robots to danger. Our approach not only performs zero-shot traversability and mitigates the risks associated with real-world data collection but also accelerates the development of advanced navigation systems, offering a cost-effective and scalable solution. To support our findings, we present navigation results, in both controlled indoor and unstructured outdoor environments. As shown in the experiments, our method provides safer navigation when compared to other state-of-the-art methods, constantly reaching the final goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16846v2">BASIL: Bayesian Assessment of Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Sycophancy (overly agreeable or flattering behavior) is critical to understand in the context of human-AI collaboration, especially in decision-making settings like health, law, and education. Existing methods for studying sycophancy in LLMs are either descriptive (study behavior change when sycophancy is elicited) or normative (provide values-based judgment on behavior change). Together, these approaches help us understand the extent, and impacts, of sycophancy. However, existing normative approaches only apply for objective tasks where ground-truth data exists, ignoring the natural subjectivity in many NLP tasks. Drawing from behavioral economics and rational decision theory, we introduce an Bayesian framework to study the normative effects of sycophancy on rationality in LLMs, without requiring labeled ground-truth. Using this interdisciplinary framework, we study sycophantic behavior in multiple LLM baselines across three different tasks, experimenting with various methods for eliciting sycophancy and obtaining probability judgments from LLMs. We find significant evidence of sycophancy in our experiments (7 of 8 baselines for one of our probing techniques), and observe that sycophancy is more likely to reduce rationality than it is to increase rationality in LLMs' decisions when they are directly probed for probabilities (2 out of 4 baselines show significant increases overall).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04652v2">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04303v2">Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 13 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Multi-agent deployments of large language models (LLMs) are increasingly embedded in market, allocation, and governance workflows, yet covert coordination among agents can silently erode trust and social welfare. Existing audits are dominated by heuristics that lack theoretical guarantees, struggle to transfer across tasks, and seldom ship with the infrastructure needed for independent replication. We introduce Audit the Whisper, a conference-grade research artifact that spans theory, benchmark design, detection, and reproducibility. Our contributions are: (i) a channel-capacity analysis showing how interventions such as paraphrase, rate limiting, and role permutation impose quantifiable capacity penalties-operationalised via paired-run Kullback--Leibler diagnostics-that tighten mutual-information thresholds with finite-sample guarantees and full proofs; (ii) ColludeBench-v0, covering pricing, first-price auctions, peer review, and hosted Gemini/Groq APIs with configurable covert schemes, deterministic manifests, and reward instrumentation; and (iii) a calibrated auditing pipeline that fuses cross-run mutual information, permutation invariance, watermark variance, and fairness-aware acceptance bias, each tuned to a $10^{-3}$ false-positive budget and validated by 10k honest runs plus an e-value martingale. Across ColludeBench and external suites including Secret Collusion, CASE, Perfect Collusion Benchmark, and SentinelAgent, the union meta-test attains state-of-the-art power at fixed FPR while ablations surface price-of-auditing trade-offs and fairness-driven colluders invisible to MI alone. We release regeneration scripts, anonymized manifests, and documentation so that external auditors can reproduce every figure, satisfy double-blind requirements, and extend the framework with minimal effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07163v4">Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16156v1">AsyncVoice Agent: Real-Time Explanation for LLM Planning and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted to the IEEE ASRU 2025 Demo Track
    </div>
    <details class="paper-abstract">
      Effective human-AI collaboration on complex reasoning tasks requires that users understand and interact with the model's process, not just receive an output. However, the monolithic text from methods like Chain-of-Thought (CoT) prevents this, as current interfaces lack real-time verbalization and robust user barge-in. We present AsyncVoice Agent, a system whose asynchronous architecture decouples a streaming LLM backend from a conversational voice frontend. This design allows narration and inference to run in parallel, empowering users to interrupt, query, and steer the model's reasoning process at any time. Objective benchmarks show this approach reduces interaction latency by more than 600x compared to monolithic baselines while ensuring high fidelity and competitive task accuracy. By enabling a two-way dialogue with a model's thought process, AsyncVoice Agent offers a new paradigm for building more effective, steerable, and trustworthy human-AI systems for high-stakes tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16147v1">Procedural Scene Programs for Open-Universe Scene Generation: LLM-Free Error Correction via Program Search</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 To appear in SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      Synthesizing 3D scenes from open-vocabulary text descriptions is a challenging, important, and recently-popular application. One of its critical subproblems is layout generation: given a set of objects, lay them out to produce a scene matching the input description. Nearly all recent work adopts a declarative paradigm for this problem: using an LLM to generate a specification of constraints between objects, then solving those constraints to produce the final layout. In contrast, we explore an alternative imperative paradigm, in which an LLM iteratively places objects, with each object's position and orientation computed as a function of previously-placed objects. The imperative approach allows for a simpler scene specification language while also handling a wider variety and larger complexity of scenes. We further improve the robustness of our imperative scheme by developing an error correction mechanism that iteratively improves the scene's validity while staying as close as possible to the original layout generated by the LLM. In forced-choice perceptual studies, participants preferred layouts generated by our imperative approach 82% and 94% of the time when compared against two declarative layout generation methods. We also present a simple, automated evaluation metric for 3D scene layout generation that aligns well with human preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16085v1">MoPHES:Leveraging on-device LLMs as Agent for Mobile Psychological Health Evaluation and Support</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      The 2022 World Mental Health Report calls for global mental health care reform, amid rising prevalence of issues like anxiety and depression that affect nearly one billion people worldwide. Traditional in-person therapy fails to meet this demand, and the situation is worsened by stigma. While general-purpose large language models (LLMs) offer efficiency for AI-driven mental health solutions, they underperform because they lack specialized fine-tuning. Existing LLM-based mental health chatbots can engage in empathetic conversations, but they overlook real-time user mental state assessment which is critical for professional counseling. This paper proposes MoPHES, a framework that integrates mental state evaluation, conversational support, and professional treatment recommendations. The agent developed under this framework uses two fine-tuned MiniCPM4-0.5B LLMs: one is fine-tuned on mental health conditions datasets to assess users' mental states and predict the severity of anxiety and depression; the other is fine-tuned on multi-turn dialogues to handle conversations with users. By leveraging insights into users' mental states, our agent provides more tailored support and professional treatment recommendations. Both models are also deployed directly on mobile devices to enhance user convenience and protect user privacy. Additionally, to evaluate the performance of MoPHES with other LLMs, we develop a benchmark for the automatic evaluation of mental state prediction and multi-turn counseling dialogues, which includes comprehensive evaluation metrics, datasets, and methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16082v1">Interpretable RNA-Seq Clustering with an LLM-Based Agentic Evidence-Grounded Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We propose CITE V.1, an agentic, evidence-grounded framework that leverages Large Language Models (LLMs) to provide transparent and reproducible interpretations of RNA-seq clusters. Unlike existing enrichment-based approaches that reduce results to broad statistical associations and LLM-only models that risk unsupported claims or fabricated citations, CITE V.1 transforms cluster interpretation by producing biologically coherent explanations explicitly anchored in the biomedical literature. The framework orchestrates three specialized agents: a Retriever that gathers domain knowledge from PubMed and UniProt, an Interpreter that formulates functional hypotheses, and Critics that evaluate claims, enforce evidence grounding, and qualify uncertainty through confidence and reliability indicators. Applied to Salmonella enterica RNA-seq data, CITE V.1 generated biologically meaningful insights supported by the literature, while an LLM-only Gemini baseline frequently produced speculative results with false citations. By moving RNA-seq analysis from surface-level enrichment to auditable, interpretable, and evidence-based hypothesis generation, CITE V.1 advances the transparency and reliability of AI in biomedicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16081v1">SARHAchat: An LLM-Based Chatbot for Sexual and Reproductive Health Counseling</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      While Artificial Intelligence (AI) shows promise in healthcare applications, existing conversational systems often falter in complex and sensitive medical domains such as Sexual and Reproductive Health (SRH). These systems frequently struggle with hallucination and lack the specialized knowledge required, particularly for sensitive SRH topics. Furthermore, current AI approaches in healthcare tend to prioritize diagnostic capabilities over comprehensive patient care and education. Addressing these gaps, this work at the UNC School of Nursing introduces SARHAchat, a proof-of-concept Large Language Model (LLM)-based chatbot. SARHAchat is designed as a reliable, user-centered system integrating medical expertise with empathetic communication to enhance SRH care delivery. Our evaluation demonstrates SARHAchat's ability to provide accurate and contextually appropriate contraceptive counseling while maintaining a natural conversational flow. The demo is available at https://sarhachat.com/}{https://sarhachat.com/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16080v1">TriAgent: Automated Biomarker Discovery with Deep Research Grounding for Triage in Acute Care by LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Emergency departments worldwide face rising patient volumes, workforce shortages, and variability in triage decisions that threaten the delivery of timely and accurate care. Current triage methods rely primarily on vital signs, routine laboratory values, and clinicians' judgment, which, while effective, often miss emerging biological signals that could improve risk prediction for infection typing or antibiotic administration in acute conditions. To address this challenge, we introduce TriAgent, a large language model (LLM)-based multi-agent framework that couples automated biomarker discovery with deep research for literature-grounded validation and novelty assessment. TriAgent employs a supervisor research agent to generate research topics and delegate targeted queries to specialized sub-agents for evidence retrieval from various data sources. Findings are synthesized to classify biomarkers as either grounded in existing knowledge or flagged as novel candidates, offering transparent justification and highlighting unexplored pathways in acute care risk stratification. Unlike prior frameworks limited to existing routine clinical biomarkers, TriAgent aims to deliver an end-to-end framework from data analysis to literature grounding to improve transparency, explainability and expand the frontier of potentially actionable clinical biomarkers. Given a user's clinical query and quantitative triage data, TriAgent achieved a topic adherence F1 score of 55.7 +/- 5.0%, surpassing the CoT-ReAct agent by over 10%, and a faithfulness score of 0.42 +/- 0.39, exceeding all baselines by more than 50%. Across experiments, TriAgent consistently outperformed state-of-the-art LLM-based agentic frameworks in biomarker justification and literature-grounded novelty assessment. We share our repo: https://github.com/CellFace/TriAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16079v1">EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at https://github.com/Edaizi/EvolveR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16062v1">Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 38 pages, 25 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: https://correctbench.github.io/
    </details>
</div>
