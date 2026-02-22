# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17622v1">What Makes a Good LLM Agent for Real-world Penetration Testing?</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      LLM-based agents show promise for automating penetration testing, yet reported performance varies widely across systems and benchmarks. We analyze 28 LLM-based penetration testing systems and evaluate five representative implementations across three benchmarks of increasing complexity. Our analysis reveals two distinct failure modes: Type A failures stem from capability gaps (missing tools, inadequate prompts) that engineering readily addresses, while Type B failures persist regardless of tooling due to planning and state management limitations. We show that Type B failures share a root cause that is largely invariant to the underlying LLM: agents lack real-time task difficulty estimation. As a result, agents misallocate effort, over-commit to low-value branches, and exhaust context before completing attack chains. Based on this insight, we present Excalibur, a penetration testing agent that couples strong tooling with difficulty-aware planning. A Tool and Skill Layer eliminates Type A failures through typed interfaces and retrieval-augmented knowledge. A Task Difficulty Assessment (TDA) mechanism addresses Type B failures by estimating tractability through four measurable dimensions (horizon estimation, evidence confidence, context load, and historical success) and uses these estimates to guide exploration-exploitation decisions within an Evidence-Guided Attack Tree Search (EGATS) framework. Excalibur achieves up to 91% task completion on CTF benchmarks with frontier models (39 to 49% relative improvement over baselines) and compromises 4 of 5 hosts on the GOAD Active Directory environment versus 2 by prior systems. These results show that difficulty-aware planning yields consistent end-to-end gains across models and addresses a limitation that model scaling alone does not eliminate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17598v1">The Cascade Equivalence Hypothesis: When Do Speech LLMs Behave Like ASR$\rightarrow$LLM Pipelines?</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 10 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Current speech LLMs largely perform implicit ASR: on tasks solvable from a transcript, they are behaviorally and mechanistically equivalent to simple Whisper$\to$LLM cascades. We show this through matched-backbone testing across four speech LLMs and six tasks, controlling for the LLM backbone for the first time. Ultravox is statistically indistinguishable from its matched cascade ($κ{=}0.93$); logit lens reveals literal text emerging in hidden states; LEACE concept erasure confirms text representations are causally necessary in both architectures tested, collapsing accuracy to near-zero. Qwen2-Audio genuinely diverges, revealing cascade equivalence is architecture-dependent, not universal. For most deployed use cases, current speech LLMs are expensive cascades, and under noise, they are worse ones, with clean-condition advantages reversing by up to 7.6% at 0 dB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22075v4">CoSpaDi: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) often relies on low-rank weight approximations that represent each column of the weight matrix in a shared low-dimensional subspace. This strategy is computationally efficient but the underlying constraint can be overly rigid for heterogeneous projection weights and may incur avoidable accuracy loss. We propose CoSpaDi (Compression via Sparse Dictionary Learning), a training-free framework that replaces low-rank factorization with a structured sparse decomposition in which each weight matrix is represented as a dense dictionary multiplied by a column-sparse coefficient matrix. This yields a union-of-subspaces model: the columns of the weight matrix are represented as linear combinations of different subsets of dictionary atoms, improving expressiveness at a fixed parameter budget. CoSpaDi is calibration-guided: using a small calibration set, we optimize the factorization to minimize functional reconstruction error of layer outputs rather than weight-space error. An activation-derived Gram orthonormalization reformulates this data-aware objective into a standard dictionary learning problem on transformed weights, and we support both per-layer compression and cross-layer dictionary sharing within groups of similar projections. Across Llama and Qwen model families, CoSpaDi consistently improves the accuracy--compression and perplexity--compression trade-offs over state-of-the-art SVD-based baselines and strong structured pruning baselines at 20-40\% compression ratios. The resulting structured sparsity enables sparse--dense computation and integrates with post-training quantization of the sparse coefficients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17560v1">ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Activation steering, or representation engineering, offers a lightweight approach to align large language models (LLMs) by manipulating their internal activations at inference time. However, current methods suffer from two key limitations: \textit{(i)} the lack of a unified theoretical framework for guiding the design of steering directions, and \textit{(ii)} an over-reliance on \textit{one-step steering} that fail to capture complex patterns of activation distributions. In this work, we propose a unified ordinary differential equations (ODEs)-based \textit{theoretical} framework for activation steering in LLM alignment. We show that conventional activation addition can be interpreted as a first-order approximation to the solution of an ODE. Based on this ODE perspective, identifying a steering direction becomes equivalent to designing a \textit{barrier function} from control theory. Derived from this framework, we introduce ODESteer, a kind of ODE-based steering guided by barrier functions, which shows \textit{empirical} advancement in LLM alignment. ODESteer identifies steering directions by defining the barrier function as the log-density ratio between positive and negative activations, and employs it to construct an ODE for \textit{multi-step and adaptive} steering. Compared to state-of-the-art activation steering methods, ODESteer achieves consistent empirical improvements on diverse LLM alignment benchmarks, a notable $5.7\%$ improvement over TruthfulQA, $2.5\%$ over UltraFeedback, and $2.4\%$ over RealToxicityPrompts. Our work establishes a principled new view of activation steering in LLM alignment by unifying its theoretical foundations via ODEs, and validating it empirically through the proposed ODESteer method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17550v1">MASPO: Unifying Gradient Utilization, Probability Mass, and Signal Reliability for Robust and Sample-Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Existing Reinforcement Learning with Verifiable Rewards (RLVR) algorithms, such as GRPO, rely on rigid, uniform, and symmetric trust region mechanisms that are fundamentally misaligned with the complex optimization dynamics of Large Language Models (LLMs). In this paper, we identify three critical challenges in these methods: (1) inefficient gradient utilization caused by the binary cutoff of hard clipping, (2) insensitive probability mass arising from uniform ratio constraints that ignore the token distribution, and (3) asymmetric signal reliability stemming from the disparate credit assignment ambiguity between positive and negative samples. To bridge these gaps, we propose Mass-Adaptive Soft Policy Optimization (MASPO), a unified framework designed to harmonize these three dimensions. MASPO integrates a differentiable soft Gaussian gating to maximize gradient utility, a mass-adaptive limiter to balance exploration across the probability spectrum, and an asymmetric risk controller to align update magnitudes with signal confidence. Extensive evaluations demonstrate that MASPO serves as a robust, all-in-one RLVR solution, significantly outperforming strong baselines. Our code is available at: https://anonymous.4open.science/r/ma1/README.md.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17547v1">KLong: Training LLM Agent for Extremely Long-horizon Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      This paper introduces KLong, an open-source LLM agent trained to solve extremely long-horizon tasks. The principle is to first cold-start the model via trajectory-splitting SFT, then scale it via progressive RL training. Specifically, we first activate basic agentic abilities of a base model with a comprehensive SFT recipe. Then, we introduce Research-Factory, an automated pipeline that generates high-quality training data by collecting research papers and constructing evaluation rubrics. Using this pipeline, we build thousands of long-horizon trajectories distilled from Claude 4.5 Sonnet (Thinking). To train with these extremely long trajectories, we propose a new trajectory-splitting SFT, which preserves early context, progressively truncates later context, and maintains overlap between sub-trajectories. In addition, to further improve long-horizon task-solving capability, we propose a novel progressive RL, which schedules training into multiple stages with progressively extended timeouts. Experiments demonstrate the superiority and generalization of KLong, as shown in Figure 1. Notably, our proposed KLong (106B) surpasses Kimi K2 Thinking (1T) by 11.28% on PaperBench, and the performance improvement generalizes to other coding benchmarks like SWE-bench Verified and MLE-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17542v1">Using LLMs for Knowledge Component-level Correctness Labeling in Open-ended Coding Problems</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Fine-grained skill representations, commonly referred to as knowledge components (KCs), are fundamental to many approaches in student modeling and learning analytics. However, KC-level correctness labels are rarely available in real-world datasets, especially for open-ended programming tasks where solutions typically involve multiple KCs simultaneously. Simply propagating problem-level correctness to all associated KCs obscures partial mastery and often leads to poorly fitted learning curves. To address this challenge, we propose an automated framework that leverages large language models (LLMs) to label KC-level correctness directly from student-written code. Our method assesses whether each KC is correctly applied and further introduces a temporal context-aware Code-KC mapping mechanism to better align KCs with individual student code. We evaluate the resulting KC-level correctness labels in terms of learning curve fit and predictive performance using the power law of practice and the Additive Factors Model. Experimental results show that our framework leads to learning curves that are more consistent with cognitive theory and improves predictive performance, compared to baselines. Human evaluation further demonstrates substantial agreement between LLM and expert annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17529v1">Enhancing Large Language Models (LLMs) for Telecom using Dynamic Knowledge Graphs and Explainable Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong potential across a variety of tasks, but their application in the telecom field remains challenging due to domain complexity, evolving standards, and specialized terminology. Therefore, general-domain LLMs may struggle to provide accurate and reliable outputs in this context, leading to increased hallucinations and reduced utility in telecom operations.To address these limitations, this work introduces KG-RAG-a novel framework that integrates knowledge graphs (KGs) with retrieval-augmented generation (RAG) to enhance LLMs for telecom-specific tasks. In particular, the KG provides a structured representation of domain knowledge derived from telecom standards and technical documents, while RAG enables dynamic retrieval of relevant facts to ground the model's outputs. Such a combination improves factual accuracy, reduces hallucination, and ensures compliance with telecom specifications.Experimental results across benchmark datasets demonstrate that KG-RAG outperforms both LLM-only and standard RAG baselines, e.g., KG-RAG achieves an average accuracy improvement of 14.3% over RAG and 21.6% over LLM-only models. These results highlight KG-RAG's effectiveness in producing accurate, reliable, and explainable outputs in complex telecom scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17520v1">When Models Ignore Definitions: Measuring Semantic Override Hallucinations in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong performance on standard digital logic and Boolean reasoning tasks, yet their reliability under locally redefined semantics remains poorly understood. In many formal settings, such as circuit specifications, examinations, and hardware documentation, operators and components are explicitly redefined within narrow scope. Correct reasoning in these contexts requires models to temporarily suppress globally learned conventions in favor of prompt-local definitions. In this work, we study a systematic failure mode we term semantic override, in which an LLM reverts to its pretrained default interpretation of operators or gate behavior despite explicit redefinition in the prompt. We also identify a related class of errors, assumption injection, where models commit to unstated hardware semantics when critical details are underspecified, rather than requesting clarification. We introduce a compact micro-benchmark of 30 logic and digital-circuit reasoning tasks designed as verifier-style traps, spanning Boolean algebra, operator overloading, redefined gates, and circuit-level semantics. Evaluating three frontier LLMs, we observe persistent noncompliance with local specifications, confident but incompatible assumptions, and dropped constraints even in elementary settings. Our findings highlight a gap between surface-level correctness and specification-faithful reasoning, motivating evaluation protocols that explicitly test local unlearning and semantic compliance in formal domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17483v1">What Do LLMs Associate with Your Name? A Human-Centered Black-Box Audit of Personal Data</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), and conversational agents based on them, are exposed to personal data (PD) during pre-training and during user interactions. Prior work shows that PD can resurface, yet users lack insight into how strongly models associate specific information to their identity. We audit PD across eight LLMs (3 open-source; 5 API-based, including GPT-4o), introduce LMP2 (Language Model Privacy Probe), a human-centered, privacy-preserving audit tool refined through two formative studies (N=20), and run two studies with EU residents to capture (i) intuitions about LLM-generated PD (N1=155) and (ii) reactions to tool output (N2=303). We show empirically that models confidently generate multiple PD categories for well-known individuals. For everyday users, GPT-4o generates 11 features with 60% or more accuracy (e.g., gender, hair color, languages). Finally, 72% of participants sought control over model-generated associations with their name, raising questions about what counts as PD and whether data privacy rights should extend to LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17481v1">ShadAR: LLM-driven shader generation to transform visual perception in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Augmented Reality (AR) can simulate various visual perceptions, such as how individuals with colorblindness see the world. However, these simulations require developers to predefine each visual effect, limiting flexibility. We present ShadAR, an AR application enabling real-time transformation of visual perception through shader generation using large language models (LLMs). ShadAR allows users to express their visual intent via natural language, which is interpreted by an LLM to generate corresponding shader code. This shader is then compiled real-time to modify the AR headset viewport. We present our LLM-driven shader generation pipeline and demonstrate its ability to transform visual perception for inclusiveness and creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17475v1">Small LLMs for Medical NLP: a Systematic Analysis of Few-Shot, Constraint Decoding, Fine-Tuning and Continual Pre-Training in Italian</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Paper Accepted at LREC 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) consistently excel in diverse medical Natural Language Processing (NLP) tasks, yet their substantial computational requirements often limit deployment in real-world healthcare settings. In this work, we investigate whether "small" LLMs (around one billion parameters) can effectively perform medical tasks while maintaining competitive accuracy. We evaluate models from three major families-Llama-3, Gemma-3, and Qwen3-across 20 clinical NLP tasks among Named Entity Recognition, Relation Extraction, Case Report Form Filling, Question Answering, and Argument Mining. We systematically compare a range of adaptation strategies, both at inference time (few-shot prompting, constraint decoding) and at training time (supervised fine-tuning, continual pretraining). Fine-tuning emerges as the most effective approach, while the combination of few-shot prompting and constraint decoding offers strong lower-resource alternatives. Our results show that small LLMs can match or even surpass larger baselines, with our best configuration based on Qwen3-1.7B achieving an average score +9.2 points higher than Qwen3-32B. We release a comprehensive collection of all the publicly available Italian medical datasets for NLP tasks, together with our top-performing models. Furthermore, we release an Italian dataset of 126M words from the Emergency Department of an Italian Hospital, and 175M words from various sources that we used for continual pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17413v1">DAVE: A Policy-Enforcing LLM Spokesperson for Secure Multi-Document Data Sharing</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      In current inter-organizational data spaces, usage policies are enforced mainly at the asset level: a whole document or dataset is either shared or withheld. When only parts of a document are sensitive, providers who want to avoid leaking protected information typically must manually redact documents before sharing them, which is costly, coarse-grained, and hard to maintain as policies or partners change. We present DAVE, a usage policy-enforcing LLM spokesperson that answers questions over private documents on behalf of a data provider. Instead of releasing documents, the provider exposes a natural language interface whose responses are constrained by machine-readable usage policies. We formalize policy-violating information disclosure in this setting, drawing on usage control and information flow security, and introduce virtual redaction: suppressing sensitive information at query time without modifying source documents. We describe an architecture for integrating such a spokesperson with Eclipse Dataspace Components and ODRL-style policies, and outline an initial provider-side integration prototype in which QA requests are routed through a spokesperson service instead of triggering raw document transfer. Our contribution is primarily architectural: we do not yet implement or empirically evaluate the full enforcement pipeline. We therefore outline an evaluation methodology to assess security, utility, and performance trade-offs under benign and adversarial querying as a basis for future empirical work on systematically governed LLM access to multi-party data spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16723v3">LLM Fingerprinting via Semantically Conditioned Watermarks</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Most LLM fingerprinting methods teach the model to respond to a few fixed queries with predefined atypical responses (keys). This memorization often does not survive common deployment steps such as finetuning or quantization, and such keys can be easily detected and filtered from LLM responses, ultimately breaking the fingerprint. To overcome these limitations we introduce LLM fingerprinting via semantically conditioned watermarks, replacing fixed query sets with a broad semantic domain, and replacing brittle atypical keys with a statistical watermarking signal diffused throughout each response. After teaching the model to watermark its responses only to prompts from a predetermined domain e.g., French language, the model owner can use queries from that domain to reliably detect the fingerprint and verify ownership. As we confirm in our thorough experimental evaluation, our fingerprint is both stealthy and robust to all common deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13110v2">SCOPE: Selective Conformal Optimized Pairwise LLM Judging</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as judges to replace costly human preference labels in pairwise evaluation. Despite their practicality, LLM judges remain prone to miscalibration and systematic biases. This paper proposes SCOPE (Selective Conformal Optimized Pairwise Evaluation), a framework for selective pairwise judging with finite-sample statistical guarantees. Under exchangeability, SCOPE calibrates an acceptance threshold such that the error rate among non-abstained judgments is at most a user-specified level $α$. To provide SCOPE with a bias-neutral uncertainty signal, we introduce Bidirectional Preference Entropy (BPE), which queries the judge under both response positions, aggregates the implied preference probabilities to enforce invariance to response order, and converts the aggregated probability into an entropy-based uncertainty score. Across MT-Bench, RewardBench, and Chatbot Arena, BPE improves uncertainty quality over standard confidence proxies, providing a stronger selection signal that enables SCOPE to consistently meet the target risk level while retaining good coverage across judge scales. In particular, at $α= 0.10$, SCOPE consistently satisfies the risk bound across all benchmarks and judge scales (empirical risk $\approx 0.097$ to $0.099$), while retaining substantial coverage, reaching $0.89$ on RewardBench with Qwen-14B and $0.98$ on RewardBench with Qwen-32B. Compared to naïve baselines, SCOPE accepts up to $2.4\times$ more judgments on MT-Bench with Qwen-7B under the same target risk constraint, demonstrating that BPE enables reliable and high-coverage LLM-based evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17410v1">Improving LLM-based Recommendation with Self-Hard Negatives from Intermediate Layers</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in recommender systems, where supervised fine-tuning (SFT) is commonly used for adaptation. Subsequent studies further introduce preference learning to incorporate negative samples into the training process. However, existing methods rely on sequence-level, offline-generated negatives, making them less discriminative and informative when adapting LLMs to recommendation tasks with large negative item spaces. To address these challenges, we propose ILRec, a novel preference fine-tuning framework for LLM-based recommendation, leveraging self-hard negative signals extracted from intermediate layers to improve preference learning. Specifically, we identify self-hard negative tokens from intermediate layers as fine-grained negative supervision that dynamically reflects the model's preference learning process. To effectively integrate these signals into training, we design a two-stage framework comprising cross-layer preference optimization and cross-layer preference distillation, enabling the model to jointly discriminate informative negatives and enhance the quality of negative signals from intermediate layers. In addition, we introduce a lightweight collaborative filtering model to assign token-level rewards for negative signals, mitigating the risk of over-penalizing false negatives. Extensive experiments on three datasets demonstrate ILRec's effectiveness in enhancing the performance of LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21493v2">Sci2Pol: Evaluating and Fine-tuning LLMs on Scientific-to-Policy Brief Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      We propose Sci2Pol-Bench and Sci2Pol-Corpus, the first benchmark and training dataset for evaluating and fine-tuning large language models (LLMs) on policy brief generation from a scientific paper. We build Sci2Pol-Bench on a five-stage taxonomy to mirror the human writing process: (i) Autocompletion, (ii) Understanding, (iii) Summarization, (iv) Generation, and (v) Verification. It features 18 tasks in multiple-choice and open-ended formats. Specifically, for the Generation stage, we show that BERTScore and ROUGE scores fail to capture the quality of brief writing, and introduce a new LLM-based evaluation metric aligned with expert judgement. Using this benchmark, we evaluate 13 leading open-source and commercial LLMs to uncover key limitations. To improve LLM performance on brief writing, we curate the Sci2Pol-Corpus for fine-tuning. We start by linking each cited scientific paper to its corresponding policy document, drawn from 5.6 million policy records. This produces 140,000 candidate pairs. We then employ an LLM-as-a-judge to filter high-quality examples, followed by in-context polishing using three expert-written samples as references. This process yields a final set of 639 new pairs. Finally, we fine-tune three models on Sci2Pol-Corpus: LLaMA-3.18B, Gemma-12B, and Gemma-27B. Fine-tuning leads to consistent performance improvements across Sci2Pol-Bench. Notably, after fine-tuning, Gemma-27B surpasses the much larger GPT-4o and DeepSeek-V3 (671B). These demonstrate the effectiveness of our corpus in bridging the gap between science and policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16777v2">DistillNote: Toward a Functional Evaluation Framework of LLM-Generated Clinical Note Summaries</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate summaries from clinical notes. However, their ability to preserve essential diagnostic information remains underexplored, which could lead to serious risks for patient care. This study introduces DistillNote, an evaluation framework for LLM summaries that targets their functional utility by applying the generated summary downstream in a complex clinical prediction task, explicitly quantifying how much prediction signal is retained. We generated over 192,000 LLM summaries from MIMIC-IV clinical notes with increasing compression rates: standard, section-wise, and distilled section-wise. Heart failure diagnosis was chosen as the prediction task, as it requires integrating a wide range of clinical signals. LLMs were fine-tuned on both the original notes and their summaries, and their diagnostic performance was compared using the AUROC metric. We contrasted DistillNote's results with evaluations from LLM-as-judge and clinicians, assessing consistency across different evaluation methods. Summaries generated by LLMs maintained a strong level of heart failure diagnostic signal despite substantial compression. Models trained on the most condensed summaries (about 20 times smaller) achieved an AUROC of 0.92, compared to 0.94 with the original note baseline (97 percent retention). Functional evaluation provided a new lens for medical summary assessment, emphasizing clinical utility as a key dimension of quality. DistillNote introduces a new scalable, task-based method for assessing the functional utility of LLM-generated clinical summaries. Our results detail compression-to-performance tradeoffs from LLM clinical summarization for the first time. The framework is designed to be adaptable to other prediction tasks and clinical domains, aiding data-driven decisions about deploying LLM summarizers in real-world healthcare settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17316v1">Same Meaning, Different Scores: Lexical and Syntactic Sensitivity in LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Accepted at LREC 2026
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has established standardized evaluation benchmarks as the primary instrument for model comparison. Yet, their reliability is increasingly questioned due to sensitivity to shallow variations in input prompts. This paper examines how controlled, truth-conditionally equivalent lexical and syntactic perturbations affect the absolute performance and relative ranking of 23 contemporary LLMs across three benchmarks: MMLU, SQuAD, and AMEGA. We employ two linguistically principled pipelines to generate meaning-preserving variations: one performing synonym substitution for lexical changes, and another using dependency parsing to determine applicable syntactic transformations. Results show that lexical perturbations consistently induce substantial, statistically significant performance degradation across nearly all models and tasks, while syntactic perturbations have more heterogeneous effects, occasionally improving results. Both perturbation types destabilize model leaderboards on complex tasks. Furthermore, model robustness did not consistently scale with model size, revealing strong task dependence. Overall, the findings suggest that LLMs rely more on surface-level lexical patterns than on abstract linguistic competence, underscoring the need for robustness testing as a standard component of LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17262v1">Quantifying and Mitigating Socially Desirable Responding in LLMs: A Desirability-Matched Graded Forced-Choice Psychometric Study</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Human self-report questionnaires are increasingly used in NLP to benchmark and audit large language models (LLMs), from persona consistency to safety and bias assessments. Yet these instruments presume honest responding; in evaluative contexts, LLMs can instead gravitate toward socially preferred answers-a form of socially desirable responding (SDR)-biasing questionnaire-derived scores and downstream conclusions. We propose a psychometric framework to quantify and mitigate SDR in questionnaire-based evaluation of LLMs. To quantify SDR, the same inventory is administered under HONEST versus FAKE-GOOD instructions, and SDR is computed as a direction-corrected standardized effect size from item response theory (IRT)-estimated latent scores. This enables comparisons across constructs and response formats, as well as against human instructed-faking benchmarks. For mitigation, we construct a graded forced-choice (GFC) Big Five inventory by selecting 30 cross-domain pairs from an item pool via constrained optimization to match desirability. Across nine instruction-tuned LLMs evaluated on synthetic personas with known target profiles, Likert-style questionnaires show consistently large SDR, whereas desirability-matched GFC substantially attenuates SDR while largely preserving the recovery of the intended persona profiles. These results highlight a model-dependent SDR-recovery trade-off and motivate SDR-aware reporting practices for questionnaire-based benchmarking and auditing of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16346v2">Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      LLM-based agents execute real-world workflows via tools and memory. These affordances enable ill-intended adversaries to also use these agents to carry out complex misuse scenarios. Existing agent misuse benchmarks largely test single-prompt instructions, leaving a gap in measuring how agents end up helping with harmful or illegal tasks over multiple turns. We introduce STING (Sequential Testing of Illicit N-step Goal execution), an automated red-teaming framework that constructs a step-by-step illicit plan grounded in a benign persona and iteratively probes a target agent with adaptive follow-ups, using judge agents to track phase completion. We further introduce an analysis framework that models multi-turn red-teaming as a time-to-first-jailbreak random variable, enabling analysis tools like discovery curves, hazard-ratio attribution by attack language, and a new metric: Restricted Mean Jailbreak Discovery. Across AgentHarm scenarios, STING yields substantially higher illicit-task completion than single-turn prompting and chat-oriented multi-turn baselines adapted to tool-using agents. In multilingual evaluations across six non-English settings, we find that attack success and illicit-task completion do not consistently increase in lower-resource languages, diverging from common chatbot findings. Overall, STING provides a practical way to evaluate and stress-test agent misuse in realistic deployment settings, where interactions are inherently multi-turn and often multilingual.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15868v2">Understanding LLM Failures: A Multi-Tape Turing Machine Analysis of Systematic Errors in Language Model Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 8 pages, 1 page appendix; v2 added Acknowledgements
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit failure modes on seemingly trivial tasks. We propose a formalisation of LLM interaction using a deterministic multi-tape Turing machine, where each tape represents a distinct component: input characters, tokens, vocabulary, model parameters, activations, probability distributions, and output text. The model enables precise localisation of failure modes to specific pipeline stages, revealing, e.g., how tokenisation obscures character-level structure needed for counting tasks. The model clarifies why techniques like chain-of-thought prompting help, by externalising computation on the output tape, while also revealing their fundamental limitations. This approach provides a rigorous, falsifiable alternative to geometric metaphors and complements empirical scaling laws with principled error analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.10361v2">Enhancing Multilingual LLM Pretraining with Model-Based Data Selection</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Dataset curation has become a basis for strong large language model (LLM) performance. While various rule-based filtering heuristics exist for English and multilingual datasets, model-based filtering techniques have primarily focused on English. To address the disparity stemming from limited research on non-English languages, we develop a model-based filtering framework for multilingual datasets that aims to identify a diverse set of structured and knowledge-rich samples. Our approach emphasizes transparency, simplicity, and efficiency, leveraging Transformer- and FastText-based classifiers to ensure the broad accessibility of our technique and data. We conduct comprehensive ablation studies on the FineWeb-2 web crawl dataset across diverse language families, scripts, and resource availability to demonstrate the effectiveness of our method. Training a 1B-parameter Llama model for 70B and 119B tokens, our approach can match the baseline MMLU score with as little as 15% of the training tokens, while also improving across other benchmarks and mitigating the curse of multilinguality. These findings provide strong evidence for the generalizability of our approach to other languages. As a result, we extend our framework to 20 languages for which we release the refined pretraining datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17234v1">All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection in LLM Backtesting</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 8 pages plus appendix
    </div>
    <details class="paper-abstract">
      To evaluate whether LLMs can accurately predict future events, we need the ability to \textit{backtest} them on events that have already resolved. This requires models to reason only with information available at a specified past date. Yet LLMs may inadvertently leak post-cutoff knowledge encoded during training, undermining the validity of retrospective evaluation. We introduce a claim-level framework for detecting and quantifying this \emph{temporal knowledge leakage}. Our approach decomposes model rationales into atomic claims and categorizes them by temporal verifiability, then applies \textit{Shapley values} to measure each claim's contribution to the prediction. This yields the \textbf{Shapley}-weighted \textbf{D}ecision-\textbf{C}ritical \textbf{L}eakage \textbf{R}ate (\textbf{Shapley-DCLR}), an interpretable metric that captures what fraction of decision-driving reasoning derives from leaked information. Building on this framework, we propose \textbf{Time}-\textbf{S}upervised \textbf{P}rediction with \textbf{E}xtracted \textbf{C}laims (\textbf{TimeSPEC}), which interleaves generation with claim verification and regeneration to proactively filter temporal contamination -- producing predictions where every supporting claim can be traced to sources available before the cutoff date. Experiments on 350 instances spanning U.S. Supreme Court case prediction, NBA salary estimation, and stock return ranking reveal substantial leakage in standard prompting baselines. TimeSPEC reduces Shapley-DCLR while preserving task performance, demonstrating that explicit, interpretable claim-level verification outperforms prompt-based temporal constraints for reliable backtesting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17229v1">Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      The black-box nature of Large Language Models necessitates novel evaluation frameworks that transcend surface-level performance metrics. This study investigates the internal neural representations of cognitive complexity using Bloom's Taxonomy as a hierarchical lens. By analyzing high-dimensional activation vectors from different LLMs, we probe whether different cognitive levels, ranging from basic recall (Remember) to abstract synthesis (Create), are linearly separable within the model's residual streams. Our results demonstrate that linear classifiers achieve approximately 95% mean accuracy across all Bloom levels, providing strong evidence that cognitive level is encoded in a linearly accessible subspace of the model's representations. These findings provide evidence that the model resolves the cognitive difficulty of a prompt early in the forward pass, with representations becoming increasingly separable across layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17223v1">Privacy-Preserving Mechanisms Enable Cheap Verifiable Inference of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, fewer users are able to host and run models locally. This has led to increased use of third-party hosting services. However, in this setting, there is a lack of guarantees on the computation performed by the inference provider. For example, a dishonest provider may replace an expensive large model with a cheaper-to-run weaker model and return the results from the weaker model to the user. Existing tools to verify inference typically rely on methods from cryptography such as zero-knowledge proofs (ZKPs), but these add significant computational overhead, and remain infeasible for use for large models. In this work, we develop a new insight -- that given a method for performing private LLM inference, one can obtain forms of verified inference at marginal extra cost. Specifically, we propose two new protocols which leverage privacy-preserving LLM inference in order to provide guarantees over the inference that was carried out. Our approaches are cheap, requiring the addition of a few extra tokens of computation, and have little to no downstream impact. As the fastest privacy-preserving inference methods are typically faster than ZK methods, the proposed protocols also improve verification runtime. Our work provides novel insights into the connections between privacy and verifiability in LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12414v2">propella-1: Multi-Property Document Annotation for LLM Data Curation at Scale</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Release: https://hf.co/collections/ellamind/propella-1
    </div>
    <details class="paper-abstract">
      Since FineWeb-Edu, data curation for LLM pretraining has predominantly relied on single scalar quality scores produced by small classifiers. A single score conflates multiple quality dimensions, prevents flexible filtering, and offers no interpretability. We introduce propella-1, a family of small multilingual LLMs (0.6B, 1.7B, 4B parameters) that annotate text documents across 18 properties organized into six categories: core content, classification, quality and value, audience and purpose, safety and compliance, and geographic relevance. The models support 57 languages and produce structured JSON annotations conforming to a predefined schema. Evaluated against a frontier commercial LLM as a reference annotator, the 4B model achieves higher agreement than much larger general-purpose models. We release propella-annotations, a dataset of over three billion document annotations covering major pretraining corpora including data from FineWeb-2, FinePDFs, HPLT 3.0, and Nemotron-CC. Using these annotations, we present a multi-dimensional compositional analysis of widely used pretraining datasets, revealing substantial differences in quality, reasoning depth, and content composition that single-score approaches cannot capture. All model weights and annotations are released under permissive, commercial-use licenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v5">Bridging Symbolic Control and Neural Reasoning in LLM Agents: Structured Cognitive Loop with a Governance Layer</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 The SCL diagram has been revised for greater clarity
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). Soft Symbolic Control constitutes a dedicated governance layer within SCL, applying symbolic constraints to probabilistic inference while preserving the flexibility of neural reasoning and restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17170v1">When LLM Judges Inflate Scores: Exploring Overrating in Relevance Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Human relevance assessment is time-consuming and cognitively intensive, limiting the scalability of Information Retrieval evaluation. This has led to growing interest in using large language models (LLMs) as proxies for human judges. However, it remains an open question whether LLM-based relevance judgments are reliable, stable, and rigorous enough to match humans for relevance assessment. In this work, we conduct a systematic study of overrating behavior in LLM-based relevance judgments across model backbones, evaluation paradigms (pointwise and pairwise), and passage modification strategies. We show that models consistently assign inflated relevance scores -- often with high confidence -- to passages that do not genuinely satisfy the underlying information need, revealing a system-wide bias rather than random fluctuations in judgment. Furthermore, controlled experiments show that LLM-based relevance judgments can be highly sensitive to passage length and surface-level lexical cues. These results raise concerns about the usage of LLMs as drop-in replacements for human relevance assessors, and highlight the urgent need for careful diagnostic evaluation frameworks when applying LLMs for relevance assessments. Our code and results are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01569v3">CaveAgent: Transforming LLMs into Stateful Runtime Operators</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 ver.2
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly capable of complex task execution, yet current agentic systems remain constrained by text-centric paradigms that struggle with long-horizon tasks due to fragile multi-turn dependencies and context drift. We present CaveAgent, a framework that shifts tool use from ``LLM-as-Text-Generator'' to ``LLM-as-Runtime-Operator.'' CaveAgent introduces a dual-stream architecture that inverts the conventional paradigm: rather than treating the LLM's text context as the primary workspace with tools as auxiliary, CaveAgent elevates the persistent Python runtime as the central locus of state, with a lightweight semantic stream serving as its orchestrator. Beyond leveraging code generation to resolve interdependent sub-tasks (e.g., loops, conditionals) in a single step, CaveAgent introduces \textit{Stateful Runtime Management}: it injects, manipulates, and retrieves complex Python objects (e.g., DataFrames, database connections) that persist across turns, unlike existing code-based approaches that remain text-bound. CaveAgent further provides a runtime-integrated skill management system that extends the Agent Skills open standard, enabling ecosystem interoperability through executable skill injections. This persistence mechanism serves as a high-fidelity external memory that reduces context drift in multi-turn interactions and preserves processed data for downstream applications without information loss. Evaluations show consistent improvement across challenging benchmarks, enabling CaveAgent to handle data scales that cause context overflow in both JSON-based and code-based agents. The accessible runtime state further provides programmatically verifiable feedback, enabling automated evaluation and reward signal generation without human annotation and establishing a structural foundation for future research in Reinforcement Learning with Verifiable Rewards (RLVR).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16241v2">Are LLMs Ready to Replace Bangla Annotators?</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as automated annotators to scale dataset creation, yet their reliability as unbiased annotators--especially for low-resource and identity-sensitive settings--remains poorly understood. In this work, we study the behavior of LLMs as zero-shot annotators for Bangla hate speech, a task where even human agreement is challenging, and annotator bias can have serious downstream consequences. We conduct a systematic benchmark of 17 LLMs using a unified evaluation framework. Our analysis uncovers annotator bias and substantial instability in model judgments. Surprisingly, increased model scale does not guarantee improved annotation quality--smaller, more task-aligned models frequently exhibit more consistent behavior than their larger counterparts. These results highlight important limitations of current LLMs for sensitive annotation tasks in low-resource languages and underscore the need for careful evaluation before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19771v3">Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16699v2">Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      LLMs are increasingly being used for complex problems which are not necessarily resolved in a single response, but require interacting with an environment to acquire information. In these scenarios, LLMs must reason about inherent cost-uncertainty tradeoffs in when to stop exploring and commit to an answer. For instance, on a programming task, an LLM should test a generated code snippet if it is uncertain about the correctness of that code; the cost of writing a test is nonzero, but typically lower than the cost of making a mistake. In this work, we show that we can induce LLMs to explicitly reason about balancing these cost-uncertainty tradeoffs, then perform more optimal environment exploration. We formalize multiple tasks, including information retrieval and coding, as sequential decision-making problems under uncertainty. Each problem has latent environment state that can be reasoned about via a prior which is passed to the LLM agent. We introduce a framework called Calibrate-Then-Act (CTA), where we feed the LLM this additional context to enable it to act more optimally. This improvement is preserved even under RL training of both the baseline and CTA. Our results on information-seeking QA and on a simplified coding task show that making cost-benefit tradeoffs explicit with CTA can help agents discover more optimal decision-making strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10117v3">Biases in the Blind Spot: Detecting What LLMs Fail to Mention</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often provide chain-of-thought (CoT) reasoning traces that appear plausible, but may hide internal biases. We call these *unverbalized biases*. Monitoring models via their stated reasoning is therefore unreliable, and existing bias evaluations typically require predefined categories and hand-crafted datasets. In this work, we introduce a fully automated, black-box pipeline for detecting task-specific unverbalized biases. Given a task dataset, the pipeline uses LLM autoraters to generate candidate bias concepts. It then tests each concept on progressively larger input samples by generating positive and negative variations, and applies statistical techniques for multiple testing and early stopping. A concept is flagged as an unverbalized bias if it yields statistically significant performance differences while not being cited as justification in the model's CoTs. We evaluate our pipeline across seven LLMs on three decision tasks (hiring, loan approval, and university admissions). Our technique automatically discovers previously unknown biases in these models (e.g., Spanish fluency, English proficiency, writing formality). In the same run, the pipeline also validates biases that were manually identified by prior work (gender, race, religion, ethnicity). More broadly, our proposed approach provides a practical, scalable path to automatic task-specific bias discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17508v4">On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Published in ICLR 2026; Project Page: https://github.com/complex-reasoning/RPG
    </div>
    <details class="paper-abstract">
      Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator ($k_1/k_2/k_3$), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (RPG) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used $k_3$ penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term; and (iv) introduces RPG-Style Clip, a clipped-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to $+6$ absolute percentage points over DAPO. We extend our experiments to 8K context length, and RPG-REINFORCE with RPG-Style Clip achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct model (47%). Notably, RPG is a stable and scalable RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) clipped importance sampling, and (c) an iterative reference-policy update scheme. Project Page: https://github.com/complex-reasoning/RPG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20650v4">FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Accurate interpretation of numerical data in financial reports is critical for markets and regulators. Although XBRL (eXtensible Business Reporting Language) provides a standard for tagging financial figures, mapping thousands of facts to over 10k US GAAP concepts remains costly and error prone. Existing benchmarks oversimplify this task as flat, single step classification over small subsets of concepts, ignoring the hierarchical semantics of the taxonomy and the structured nature of financial documents. Consequently, these benchmarks fail to evaluate Large Language Models (LLMs) under realistic reporting conditions. To bridge this gap, we introduce FinTagging, the first comprehensive benchmark for structure aware and full scope XBRL tagging. We decompose the complex tagging process into two subtasks: (1) FinNI (Financial Numeric Identification), which extracts entities and types from heterogeneous contexts including text and tables; and (2) FinCL (Financial Concept Linking), which maps extracted entities to the full US GAAP taxonomy. This two stage formulation enables a fair assessment of LLMs' capabilities in numerical reasoning and taxonomy alignment. Evaluating diverse LLMs in zero shot settings reveals that while models generalize well in extraction, they struggle significantly with fine grained concept linking, highlighting critical limitations in domain specific structure aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08886v2">FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Going beyond simple text processing, financial auditing requires detecting semantic, structural, and numerical inconsistencies across large-scale disclosures. As financial reports are filed in XBRL, a structured XML format governed by accounting standards, auditing becomes a structured information extraction and reasoning problem involving concept alignment, taxonomy-defined relations, and cross-document consistency. Although large language models (LLMs) show promise on isolated financial tasks, their capability in professional-grade auditing remains unclear. We introduce FinAuditing, a taxonomy-aligned, structure-aware benchmark built from real XBRL filings. It contains 1,102 annotated instances averaging over 33k tokens and defines three tasks: Financial Semantic Matching (FinSM), Financial Relationship Extraction (FinRE), and Financial Mathematical Reasoning (FinMR). Evaluations of 13 state-of-the-art LLMs reveal substantial gaps in concept retrieval, taxonomy-aware relation modeling, and consistent cross-document reasoning. These findings highlight the need for realistic, structure-aware benchmarks. We release the evaluation code at https://github.com/The-FinAI/FinAuditing and the dataset at https://huggingface.co/collections/TheFinAI/finauditing. The task currently serves as the official benchmark of an ongoing public evaluation contest at https://open-finance-lab.github.io/SecureFinAI_Contest_2026/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16997v1">Exploring LLMs for User Story Extraction from Mockups</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 14 pages, 6 figures. Preprint of the paper published in the 28th Workshop on Requirements Engineering (WER 2025)
    </div>
    <details class="paper-abstract">
      User stories are one of the most widely used artifacts in the software industry to define functional requirements. In parallel, the use of high-fidelity mockups facilitates end-user participation in defining their needs. In this work, we explore how combining these techniques with large language models (LLMs) enables agile and automated generation of user stories from mockups. To this end, we present a case study that analyzes the ability of LLMs to extract user stories from high-fidelity mockups, both with and without the inclusion of a glossary of the Language Extended Lexicon (LEL) in the prompts. Our results demonstrate that incorporating the LEL significantly enhances the accuracy and suitability of the generated user stories. This approach represents a step forward in the integration of AI into requirements engineering, with the potential to improve communication between users and developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08351v2">The Chicken and Egg Dilemma: Co-optimizing Data and Model Configurations for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Co-optimizing data and model configurations for training LLMs presents a classic chicken-and-egg dilemma: The best training data configuration (e.g., data mixture) for a downstream task depends on the chosen model configuration (e.g., model architecture), and vice versa. However, jointly optimizing both data and model configurations is often deemed intractable, and existing methods focus on either data or model optimization without considering their interaction. We introduce JoBS, an approach that uses a scaling-law-inspired performance predictor to aid Bayesian optimization (BO) in jointly optimizing LLM training data and model configurations efficiently. JoBS allocates a portion of the optimization budget to learn an LLM performance predictor that predicts how promising a training configuration is from a small number of training steps. The remaining budget is used to perform BO entirely with the predictor, effectively amortizing the cost of running full-training runs. We study JoBS's average regret and devise the optimal budget allocation to minimize regret. JoBS outperforms existing multi-fidelity BO baselines, as well as data and model optimization approaches across diverse LLM tasks under the same optimization budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16943v1">Mind the GAP: Text Safety Does Not Transfer to Tool-Call Safety in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 23 pages, 5 figures, 4 tables, code and data at https://github.com/acartag7/gap-benchmark
    </div>
    <details class="paper-abstract">
      Large language models deployed as agents increasingly interact with external systems through tool calls--actions with real-world consequences that text outputs alone do not carry. Safety evaluations, however, overwhelmingly measure text-level refusal behavior, leaving a critical question unanswered: does alignment that suppresses harmful text also suppress harmful actions? We introduce the GAP benchmark, a systematic evaluation framework that measures divergence between text-level safety and tool-call-level safety in LLM agents. We test six frontier models across six regulated domains (pharmaceutical, financial, educational, employment, legal, and infrastructure), seven jailbreak scenarios per domain, three system prompt conditions (neutral, safety-reinforced, and tool-encouraging), and two prompt variants, producing 17,420 analysis-ready datapoints. Our central finding is that text safety does not transfer to tool-call safety. Across all six models, we observe instances where the model's text output refuses a harmful request while its tool calls simultaneously execute the forbidden action--a divergence we formalize as the GAP metric. Even under safety-reinforced system prompts, 219 such cases persist across all six models. System prompt wording exerts substantial influence on tool-call behavior: TC-safe rates span 21 percentage points for the most robust model and 57 for the most prompt-sensitive, with 16 of 18 pairwise ablation comparisons remaining significant after Bonferroni correction. Runtime governance contracts reduce information leakage in all six models but produce no detectable deterrent effect on forbidden tool-call attempts themselves. These results demonstrate that text-only safety evaluations are insufficient for assessing agent behavior and that tool-call safety requires dedicated measurement and mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16935v1">DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 18 Pages, 7 Tables, 1 Figure
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) capabilities have scaled, safety guardrails remain largely stateless, treating multi-turn dialogues as a series of disconnected events. This lack of temporal awareness facilitates a "Safety Gap" where adversarial tactics, like Crescendo and ActorAttack, slowly bleed malicious intent across turn boundaries to bypass stateless filters. We introduce DeepContext, a stateful monitoring framework designed to map the temporal trajectory of user intent. DeepContext discards the isolated evaluation model in favor of a Recurrent Neural Network (RNN) architecture that ingests a sequence of fine-tuned turn-level embeddings. By propagating a hidden state across the conversation, DeepContext captures the incremental accumulation of risk that stateless models overlook. Our evaluation demonstrates that DeepContext significantly outperforms existing baselines in multi-turn jailbreak detection, achieving a state-of-the-art F1 score of 0.84, which represents a substantial improvement over both hyperscaler cloud-provider guardrails and leading open-weight models such as Llama-Prompt-Guard-2 (0.67) and Granite-Guardian (0.67). Furthermore, DeepContext maintains a sub-20ms inference overhead on a T4 GPU, ensuring viability for real-time applications. These results suggest that modeling the sequential evolution of intent is a more effective and computationally efficient alternative to deploying massive, stateless models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16932v1">RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Retrieval algorithms like BM25 and query likelihood with Dirichlet smoothing remain strong and efficient first-stage rankers, yet improvements have mostly relied on parameter tuning and human intuition. We investigate whether a large language model, guided by an evaluator and evolutionary search, can automatically discover improved lexical retrieval algorithms. We introduce RankEvolve, a program evolution setup based on AlphaEvolve, in which candidate ranking algorithms are represented as executable code and iteratively mutated, recombined, and selected based on retrieval performance across 12 IR datasets from BEIR and BRIGHT. RankEvolve starts from two seed programs: BM25 and query likelihood with Dirichlet smoothing. The evolved algorithms are novel, effective, and show promising transfer to the full BEIR and BRIGHT benchmarks as well as TREC DL 19 and 20. Our results suggest that evaluator-guided LLM program evolution is a practical path towards automatic discovery of novel ranking algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06275v2">RoPE-LIME: RoPE-Space Locality + Sparse-K Sampling for Efficient LLM Attribution</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Explaining closed-source Large Language Model (LLM) outputs is challenging because API access prevents gradient-based attribution, while perturbation methods are costly and noisy when they depend on regenerated text. We introduce \textbf{Rotary Positional Embedding Linear Local Interpretable Model-agnostic Explanations (RoPE-LIME)}, an open-source extension of gSMILE that decouples reasoning from explanation: given a fixed output from a closed model, a smaller open-source surrogate computes token-level attributions from probability-based objectives (negative log-likelihood and divergence targets) under input perturbations. RoPE-LIME incorporates (i) a locality kernel based on Relaxed Word Mover's Distance computed in \textbf{RoPE embedding space} for stable similarity under masking, and (ii) \textbf{Sparse-$K$} sampling, an efficient perturbation strategy that improves interaction coverage under limited budgets. Experiments on HotpotQA (sentence features) and a hand-labeled MMLU subset (word features) show that RoPE-LIME produces more informative attributions than leave-one-out sampling and improves over gSMILE while substantially reducing closed-model API calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16902v1">LLM-WikiRace: Benchmarking Long-term Planning and Reasoning over Real-World Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      We introduce LLM-Wikirace, a benchmark for evaluating planning, reasoning, and world knowledge in large language models (LLMs). In LLM-Wikirace, models must efficiently navigate Wikipedia hyperlinks step by step to reach a target page from a given source, requiring look-ahead planning and the ability to reason about how concepts are connected in the real world. We evaluate a broad set of open- and closed-source models, including Gemini-3, GPT-5, and Claude Opus 4.5, which achieve the strongest results on the easy level of the task and demonstrate superhuman performance. Despite this, performance drops sharply on hard difficulty: the best-performing model, Gemini-3, succeeds in only 23\% of hard games, highlighting substantial remaining challenges for frontier models. Our analysis shows that world knowledge is a necessary ingredient for success, but only up to a point, beyond this threshold, planning and long-horizon reasoning capabilities become the dominant factors. Trajectory-level analysis further reveals that even the strongest models struggle to replan after failure, frequently entering loops rather than recovering. LLM-Wikirace is a simple benchmark that reveals clear limitations in current reasoning systems, offering an open arena where planning-capable LLMs still have much to prove. Our code and leaderboard available at https:/llmwikirace.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16901v1">AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly deployed in long-horizon, complex environments to solve challenging problems, but this expansion exposes them to long-horizon attacks that exploit multi-turn user-agent-environment interactions to achieve objectives infeasible in single-turn settings. To measure agent vulnerabilities to such risks, we present AgentLAB, the first benchmark dedicated to evaluating LLM agent susceptibility to adaptive, long-horizon attacks. Currently, AgentLAB supports five novel attack types including intent hijacking, tool chaining, task injection, objective drifting, and memory poisoning, spanning 28 realistic agentic environments, and 644 security test cases. Leveraging AgentLAB, we evaluate representative LLM agents and find that they remain highly susceptible to long-horizon attacks; moreover, defenses designed for single-turn interactions fail to reliably mitigate long-horizon threats. We anticipate that AgentLAB will serve as a valuable benchmark for tracking progress on securing LLM agents in practical settings. The benchmark is publicly available at https://tanqiujiang.github.io/AgentLAB_main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07287v2">Patch-to-PoC: A Systematic Study of Agentic LLM Systems for Linux Kernel N-Day Reproduction</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 17 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Autonomous large language model (LLM) based systems have recently shown promising results across a range of cybersecurity tasks. However, there is no systematic study on their effectiveness in autonomously reproducing Linux kernel vulnerabilities with concrete proofs-of-concept (PoCs). Owing to the size, complexity, and low-level nature of the Linux kernel, such tasks are widely regarded as particularly challenging for current LLM-based approaches. In this paper, we present the first large-scale study of LLM-based Linux kernel vulnerability reproduction. For this purpose, we develop K-Repro, an LLM-based agentic system equipped with controlled code-browsing, virtual machine management, interaction, and debugging capabilities. Using kernel security patches as input, K-Repro automates end-to-end bug reproduction of N-day vulnerabilities in the Linux kernel. On a dataset of 100 real-world exploitable Linux kernel vulnerabilities collected from KernelCTF, our results show that K-Repro can generate PoCs that reproduce over 50\% of the cases with practical time and monetary cost. Beyond aggregate success rates, we perform an extensive study of effectiveness, efficiency, stability, and impact factors to explain when agentic reproduction succeeds, where it fails, and which components drive performance. These findings provide actionable guidance for building more reliable autonomous security agents and for assessing real-world N-day risk from both offensive and defensive perspectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16835v1">NeST: Neuron Selective Tuning for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Safety alignment is essential for the responsible deployment of large language models (LLMs). Yet, existing approaches often rely on heavyweight fine-tuning that is costly to update, audit, and maintain across model families. Full fine-tuning incurs substantial computational and storage overhead, while parameter-efficient methods such as LoRA trade efficiency for inconsistent safety gains and sensitivity to design choices. Safety intervention mechanisms such as circuit breakers reduce unsafe outputs without modifying model weights, but do not directly shape or preserve the internal representations that govern safety behavior. These limitations hinder rapid and reliable safety updates, particularly in settings where models evolve frequently or must adapt to new policies and domains. We present NeST, a lightweight, structure-aware safety alignment framework that strengthens refusal behavior by selectively adapting a small subset of safety-relevant neurons while freezing the remainder of the model. NeST aligns parameter updates with the internal organization of safety behavior by clustering functionally coherent safety neurons and enforcing shared updates within each cluster, enabling targeted and stable safety adaptation without broad model modification or inference-time overhead. We benchmark NeST against three dominant baselines: full fine-tuning, LoRA-based fine-tuning, and circuit breakers across 10 open-weight LLMs spanning multiple model families and sizes. Across all evaluated models, NeST reduces the attack success rate from an average of 44.5% to 4.36%, corresponding to a 90.2% reduction in unsafe generations, while requiring only 0.44 million trainable parameters on average. This amounts to a 17,310x decrease in updated parameters compared to full fine-tuning and a 9.25x reduction relative to LoRA, while consistently achieving stronger safety performance for alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v6">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations at eBay</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering Search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, Search, and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy for human judgments. We present an empirical study on a major e-commerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases and provide more diverse keyphrase recommendations while aligning advertising, search and buyer preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16802v1">References Improve LLM Alignment in Non-Verifiable Domains</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 ICLR 2026 Camera Ready
    </div>
    <details class="paper-abstract">
      While Reinforcement Learning with Verifiable Rewards (RLVR) has shown strong effectiveness in reasoning tasks, it cannot be directly applied to non-verifiable domains lacking ground-truth verifiers, such as LLM alignment. In this work, we investigate whether reference-guided LLM-evaluators can bridge this gap by serving as soft "verifiers". First, we design evaluation protocols that enhance LLM-based evaluators for LLM alignment using reference outputs. Through comprehensive experiments, we show that a reference-guided approach substantially improves the accuracy of less capable LLM-judges using references from frontier models; stronger LLM-judges can also be enhanced by high-quality (i.e., human-written) references. Building on these improved judges, we demonstrate the utility of high-quality references in alignment tuning, where LLMs guided with references are used as judges to self-improve. We show that reference-guided self-improvement yields clear gains over both direct SFT on reference outputs and self-improvement with reference-free judges, achieving performance comparable to training with ArmoRM, a strong finetuned reward model. Specifically, our method achieves 73.1% and 58.7% on AlpacaEval and Arena-Hard with Llama-3-8B-Instruct, and 70.0% and 74.1% with Qwen2.5-7B, corresponding to average absolute gains of +20.2 / +17.1 points over SFT distillation and +5.3 / +3.6 points over reference-free self-improvement on AlpacaEval / Arena-Hard. These results highlight the potential of using reference-guided LLM-evaluators to enable effective LLM post-training in non-verifiable domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16800v1">Large-scale online deanonymization with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      We show that large language models can be used to perform at-scale deanonymization. With full Internet access, our agent can re-identify Hacker News users and Anthropic Interviewer participants at high precision, given pseudonymous online profiles and conversations alone, matching what would take hours for a dedicated human investigator. We then design attacks for the closed-world setting. Given two databases of pseudonymous individuals, each containing unstructured text written by or about that individual, we implement a scalable attack pipeline that uses LLMs to: (1) extract identity-relevant features, (2) search for candidate matches via semantic embeddings, and (3) reason over top candidates to verify matches and reduce false positives. Compared to prior deanonymization work (e.g., on the Netflix prize) that required structured data or manual feature engineering, our approach works directly on raw user content across arbitrary platforms. We construct three datasets with known ground-truth data to evaluate our attacks. The first links Hacker News to LinkedIn profiles, using cross-platform references that appear in the profiles. Our second dataset matches users across Reddit movie discussion communities; and the third splits a single user's Reddit history in time to create two pseudonymous profiles to be matched. In each setting, LLM-based methods substantially outperform classical baselines, achieving up to 68% recall at 90% precision compared to near 0% for the best non-LLM method. Our results show that the practical obscurity protecting pseudonymous users online no longer holds and that threat models for online privacy need to be reconsidered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16703v1">Measuring Mid-2025 LLM-Assistance on Novice Performance in Biology</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform strongly on biological benchmarks, raising concerns that they may help novice actors acquire dual-use laboratory skills. Yet, whether this translates to improved human performance in the physical laboratory remains unclear. To address this, we conducted a pre-registered, investigator-blinded, randomized controlled trial (June-August 2025; n = 153) evaluating whether LLMs improve novice performance in tasks that collectively model a viral reverse genetics workflow. We observed no significant difference in the primary endpoint of workflow completion (5.2% LLM vs. 6.6% Internet; P = 0.759), nor in the success rate of individual tasks. However, the LLM arm had numerically higher success rates in four of the five tasks, most notably for the cell culture task (68.8% LLM vs. 55.3% Internet; P = 0.059). Post-hoc Bayesian modeling of pooled data estimates an approximate 1.4-fold increase (95% CrI 0.74-2.62) in success for a "typical" reverse genetics task under LLM assistance. Ordinal regression modelling suggests that participants in the LLM arm were more likely to progress through intermediate steps across all tasks (posterior probability of a positive effect: 81%-96%). Overall, mid-2025 LLMs did not substantially increase novice completion of complex laboratory procedures but were associated with a modest performance benefit. These results reveal a gap between in silico benchmarks and real-world utility, underscoring the need for physical-world validation of AI biosecurity assessments as model capabilities and user proficiency evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18825v4">EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 v3 was a major revision with updated experiments and analysis; v4 consists of minor edits
    </div>
    <details class="paper-abstract">
      We develop evaluation methods for measuring the economic decision-making capabilities and tendencies of LLMs. First, we develop benchmarks derived from key problems in economics -- procurement, scheduling, and pricing -- that test an LLM's ability to learn from the environment in context. Second, we develop the framework of litmus tests, evaluations that quantify an LLM's choice behavior on a stylized decision-making task with multiple conflicting objectives. Each litmus test outputs a litmus score, which quantifies an LLM's tradeoff response, a reliability score, which measures the coherence of an LLM's choice behavior, and a competency score, which measures an LLM's capability at the same task when the conflicting objectives are replaced by a single, well-specified objective. Evaluating a broad array of frontier LLMs, we (1) investigate changes in LLM capabilities and tendencies over time, (2) derive economically meaningful insights from the LLMs' choice behavior and chain-of-thought, (3) validate our litmus test framework by testing self-consistency, robustness, and generalizability. Overall, this work provides a foundation for evaluating LLM agents as they are further integrated into economic decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16662v1">Evaluating Collective Behaviour of Hundreds of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      As autonomous agents powered by LLM are increasingly deployed in society, understanding their collective behaviour in social dilemmas becomes critical. We introduce an evaluation framework where LLMs generate strategies encoded as algorithms, enabling inspection prior to deployment and scaling to populations of hundreds of agents -- substantially larger than in previous work. We find that more recent models tend to produce worse societal outcomes compared to older models when agents prioritise individual gain over collective benefits. Using cultural evolution to model user selection of agents, our simulations reveal a significant risk of convergence to poor societal equilibria, particularly when the relative benefit of cooperation diminishes and population sizes increase. We release our code as an evaluation suite for developers to assess the emergent collective behaviour of their models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16660v1">Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      The widespread deployment of large language models (LLMs) across linguistic communities necessitates reliable multilingual safety alignment. However, recent efforts to extend alignment to other languages often require substantial resources, either through large-scale, high-quality supervision in the target language or through pairwise alignment with high-resource languages, which limits scalability. In this work, we propose a resource-efficient method for improving multilingual safety alignment. We introduce a plug-and-play Multi-Lingual Consistency (MLC) loss that can be integrated into existing monolingual alignment pipelines. By improving collinearity between multilingual representation vectors, our method encourages directional consistency at the multilingual semantic level in a single update. This allows simultaneous alignment across multiple languages using only multilingual prompt variants without requiring additional response-level supervision in low-resource languages. We validate the proposed method across different model architectures and alignment paradigms, and demonstrate its effectiveness in enhancing multilingual safety with limited impact on general model utility. Further evaluation across languages and tasks indicates improved cross-lingual generalization, suggesting the proposed approach as a practical solution for multilingual consistency alignment under limited supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15238v2">Closing the Distribution Gap in Adversarial Training for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Adversarial training for LLMs is one of the most promising methods to reliably improve robustness against adversaries. However, despite significant progress, models remain vulnerable to simple in-distribution exploits, such as rewriting prompts in the past tense or translating them into other languages. We argue that this persistent fragility stems from a fundamental limitation in current adversarial training algorithms: they minimize adversarial loss on their training set but inadequately cover the data distribution, resulting in vulnerability to seemingly simple attacks. To bridge this gap, we propose Distributional Adversarial Training, DAT. We leverage Diffusion LLMs to approximate the true joint distribution of prompts and responses, enabling generation of diverse, high-likelihood samples that address generalization failures. By combining optimization over the data distribution provided by the diffusion model with continuous adversarial training, DAT achieves substantially higher adversarial robustness than previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16610v1">Who can we trust? LLM-as-a-jury for Comparative Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied as automatic evaluators for natural language generation assessment often using pairwise comparative judgements. Existing approaches typically rely on single judges or aggregate multiple judges assuming equal reliability. In practice, LLM judges vary substantially in performance across tasks and aspects, and their judgment probabilities may be biased and inconsistent. Furthermore, human-labelled supervision for judge calibration may be unavailable. We first empirically demonstrate that inconsistencies in LLM comparison probabilities exist and show that it limits the effectiveness of direct probability-based ranking. To address this, we study the LLM-as-a-jury setting and propose BT-sigma, a judge-aware extension of the Bradley-Terry model that introduces a discriminator parameter for each judge to jointly infer item rankings and judge reliability from pairwise comparisons alone. Experiments on benchmark NLG evaluation datasets show that BT-sigma consistently outperforms averaging-based aggregation methods, and that the learned discriminator strongly correlates with independent measures of the cycle consistency of LLM judgments. Further analysis reveals that BT-sigma can be interpreted as an unsupervised calibration mechanism that improves aggregation by modelling judge reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16603v1">FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The growing demand for large language models (LLMs) requires serving systems to handle many concurrent requests with diverse service level objectives (SLOs). This exacerbates head-of-line (HoL) blocking during the compute-intensive prefill phase, where long-running requests monopolize resources and delay higher-priority ones, leading to widespread time-to-first-token (TTFT) SLO violations. While chunked prefill enables interruptibility, it introduces an inherent trade-off between responsiveness and throughput: reducing chunk size improves response latency but degrades computational efficiency, whereas increasing chunk size maximizes throughput but exacerbates blocking. This necessitates an adaptive preemption mechanism. However, dynamically balancing execution granularity against scheduling overheads remains a key challenge. In this paper, we propose FlowPrefill, a TTFT-goodput-optimized serving system that resolves this conflict by decoupling preemption granularity from scheduling frequency. To achieve adaptive prefill scheduling, FlowPrefill introduces two key innovations: 1) Operator-Level Preemption, which leverages operator boundaries to enable fine-grained execution interruption without the efficiency loss associated with fixed small chunking; and 2) Event-Driven Scheduling, which triggers scheduling decisions only upon request arrival or completion events, thereby supporting efficient preemption responsiveness while minimizing control-plane overhead. Evaluation on real-world production traces shows that FlowPrefill improves maximum goodput by up to 5.6$\times$ compared to state-of-the-art systems while satisfying heterogeneous SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23350v2">Validating Formal Specifications with LLM-generated Test Cases</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Validation is a central activity when developing formal specifications. Similarly to coding, a possible validation technique is to define upfront test cases or scenarios that a future specification should satisfy or not. Unfortunately, specifying such test cases is burdensome and error prone, which could cause users to skip this validation task. This paper reports the results of an empirical evaluation of using pre-trained large language models (LLMs) to automate the generation of test cases from natural language requirements. In particular, we focus on test cases for structural requirements of simple domain models formalized in the Alloy specification language. Our evaluation focuses on the state-of-the-art GPT-5 model, but results from other closed- and open-source LLMs are also reported. The results show that, in this context, GPT-5 is already quite effective at generating positive (and negative) test cases that are syntactically correct and that satisfy (or not) the given requirement, and that can detect many wrong specifications written by humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16516v1">Supercharging Agenda Setting Research: The ParlaCAP Dataset of 28 European Parliaments and a Scalable Multilingual LLM-Based Classification</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 17 pages, 7 figures, 7 tables. Submitted to the PoliticalNLP 2026 workshop, co-located with LREC 2026 conference
    </div>
    <details class="paper-abstract">
      This paper introduces ParlaCAP, a large-scale dataset for analyzing parliamentary agenda setting across Europe, and proposes a cost-effective method for building domain-specific policy topic classifiers. Applying the Comparative Agendas Project (CAP) schema to the multilingual ParlaMint corpus of over 8 million speeches from 28 parliaments of European countries and autonomous regions, we follow a teacher-student framework in which a high-performing large language model (LLM) annotates in-domain training data and a multilingual encoder model is fine-tuned on these annotations for scalable data annotation. We show that this approach produces a classifier tailored to the target domain. Agreement between the LLM and human annotators is comparable to inter-annotator agreement among humans, and the resulting model outperforms existing CAP classifiers trained on manually-annotated but out-of-domain data. In addition to the CAP annotations, the ParlaCAP dataset offers rich speaker and party metadata, as well as sentiment predictions coming from the ParlaSent multilingual transformer model, enabling comparative research on political attention and representation across countries. We illustrate the analytical potential of the dataset with three use cases, examining the distribution of parliamentary attention across policy topics, sentiment patterns in parliamentary speech, and gender differences in policy attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15001v2">Boundary Point Jailbreaking of Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Frontier LLMs are safeguarded against attempts to extract harmful information via adversarial prompts known as "jailbreaks". Recently, defenders have developed classifier-based systems that have survived thousands of hours of human red teaming. We introduce Boundary Point Jailbreaking (BPJ), a new class of automated jailbreak attacks that evade the strongest industry-deployed safeguards. Unlike previous attacks that rely on white/grey-box assumptions (such as classifier scores or gradients) or libraries of existing jailbreaks, BPJ is fully black-box and uses only a single bit of information per query: whether or not the classifier flags the interaction. To achieve this, BPJ addresses the core difficulty in optimising attacks against robust real-world defences: evaluating whether a proposed modification to an attack is an improvement. Instead of directly trying to learn an attack for a target harmful string, BPJ converts the string into a curriculum of intermediate attack targets and then actively selects evaluation points that best detect small changes in attack strength ("boundary points"). We believe BPJ is the first fully automated attack algorithm that succeeds in developing universal jailbreaks against Constitutional Classifiers, as well as the first automated attack algorithm that succeeds against GPT-5's input classifier without relying on human attack seeds. BPJ is difficult to defend against in individual interactions but incurs many flags during optimisation, suggesting that effective defence requires supplementing single-interaction methods with batch-level monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02515v2">PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      This paper presents a systematic investigation into the constrained generation capabilities of large language models (LLMs) in producing Songci, a classical Chinese poetry form characterized by strict structural, tonal, and rhyme constraints defined by Cipai templates. We first develop a comprehensive, multi-faceted evaluation framework that includes: (i) a formal conformity score, (ii) automated quality assessment using LLMs, (iii) human evaluation, and (iv) classification-based probing tasks. Using this framework, we evaluate the generative performance of 18 LLMs, including 3 proprietary models and 15 open-source models across 4 families, under five prompting strategies: zero-shot, one-shot, completion-based, instruction-based, and chain-of-thought. Finally, we propose a Generate-Critic architecture in which the evaluation framework functions as an automated critic. Leveraging the critic's feedback as a scoring function for best-of-N selection, we fine-tune 3 lightweight open-source LLMs via supervised fine-tuning (SFT), resulting in improvements of up to 5.88% in formal conformity. Our findings offer new insights into the generative strengths and limitations of LLMs in producing culturally significant and formally constrained literary texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16438v1">Intra-Fairness Dynamics: The Bias Spillover Effect in Targeted LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 Submitted to the BiAlign CHI Workshop 2026
    </div>
    <details class="paper-abstract">
      Conventional large language model (LLM) fairness alignment largely focuses on mitigating bias along single sensitive attributes, overlooking fairness as an inherently multidimensional and context-specific value. This approach risks creating systems that achieve narrow fairness metrics while exacerbating disparities along untargeted attributes, a phenomenon known as bias spillover. While extensively studied in machine learning, bias spillover remains critically underexplored in LLM alignment. In this work, we investigate how targeted gender alignment affects fairness across nine sensitive attributes in three state-of-the-art LLMs (Mistral 7B, Llama 3.1 8B, Qwen 2.5 7B). Using Direct Preference Optimization and the BBQ benchmark, we evaluate fairness under ambiguous and disambiguous contexts. Our findings reveal noticeable bias spillover: while aggregate results show improvements, context-aware analysis exposes significant degradations in ambiguous contexts, particularly for physical appearance ($p< 0.001$ across all models), sexual orientation, and disability status. We demonstrate that improving fairness along one attribute can inadvertently worsen disparities in others under uncertainty, highlighting the necessity of context-aware, multi-attribute fairness evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00663v2">SEISMO: Increasing Sample Efficiency in Molecular Optimization with a Trajectory-Aware LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 Fabian P. Krüger and Andrea Hunklinger contributed equally to this work
    </div>
    <details class="paper-abstract">
      Optimizing the structure of molecules to achieve desired properties is a central bottleneck across the chemical sciences, particularly in the pharmaceutical industry where it underlies the discovery of new drugs. Since molecular property evaluation often relies on costly and rate-limited oracles, such as experimental assays, molecular optimization must be highly sample-efficient. To address this, we introduce SEISMO, an LLM agent that performs strictly online, inference-time molecular optimization, updating after every oracle call without the need for population-based or batched learning. SEISMO conditions each proposal on the full optimization trajectory, combining natural-language task descriptions with scalar scores and, when available, structured explanatory feedback. Across the Practical Molecular Optimization benchmark of 23 tasks, SEISMO achieves a 2-3 times higher area under the optimisation curve than prior methods, often reaching near-maximal task scores within 50 oracle calls. Our additional medicinal-chemistry tasks show that providing explanatory feedback further improves efficiency, demonstrating that leveraging domain knowledge and structured information is key to sample-efficient molecular optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21190v2">The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 under review
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral to computing infrastructure, safety alignment serves as the primary security control preventing the generation of harmful payloads. However, this defense remains brittle. Existing jailbreak attacks typically bifurcate into white-box methods, which are inapplicable to commercial APIs due to lack of gradient access, and black-box optimization techniques, which often yield unnatural (e.g., syntactically rigid) or non-transferable (e.g., lacking cross-model generalization) prompts. In this work, we introduce TrojFill, a black-box exploitation framework that bypasses safety filters by targeting a fundamental logic flaw in current alignment paradigms: the decoupling of unsafety reasoning from content generation. TrojFill structurally reframes malicious instructions as a template-filling task required for safety analysis. By embedding obfuscated payloads (e.g., via placeholder substitution) into a "Trojan" structure, the attack induces the model to generate prohibited content as a "demonstrative example" ostensibly required for a subsequent sentence-by-sentence safety critique. This approach effectively masks the malicious intent from standard intent classifiers. We evaluate TrojFill against representative commercial systems, including GPT-4o, Gemini-2.5, DeepSeek-3.1, and Qwen-Max. Our results demonstrate that TrojFill achieves near-universal bypass rates: reaching 100% Attack Success Rate (ASR) on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o, significantly outperforming existing black-box baselines. Furthermore, unlike optimization-based adversarial prompts, TrojFill generates highly interpretable and transferable attack vectors, exposing a systematic vulnerability inaligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16304v1">Mind the Gap: Evaluating LLMs for High-Level Malicious Package Detection vs. Fine-Grained Indicator Identification</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      The prevalence of malicious packages in open-source repositories, such as PyPI, poses a critical threat to the software supply chain. While Large Language Models (LLMs) have emerged as a promising tool for automated security tasks, their effectiveness in detecting malicious packages and indicators remains underexplored. This paper presents a systematic evaluation of 13 LLMs for detecting malicious software packages. Using a curated dataset of 4,070 packages (3,700 benign and 370 malicious), we evaluate model performance across two tasks: binary classification (package detection) and multi-label classification (identification of specific malicious indicators). We further investigate the impact of prompting strategies, temperature settings, and model specifications on detection accuracy. We find a significant "granularity gap" in LLMs' capabilities. While GPT-4.1 achieves near-perfect performance in binary detection (F1 $\approx$ 0.99), performance degrades by approximately 41\% when the task shifts to identifying specific malicious indicators. We observe that general models are best for filtering out the majority of threats, while specialized coder models are better at detecting attacks that follow a strict, predictable code structure. Our correlation analysis indicates that parameter size and context width have negligible explanatory power regarding detection accuracy. We conclude that while LLMs are powerful detectors at the package level, they lack the semantic depth required for precise identification at the granular indicator level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16246v1">Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Interactive large language model (LLM) agents operating via multi-turn dialogue and multi-step tool calling are increasingly used in production. Benchmarks for these agents must both reliably compare models and yield on-policy training data. Prior agentic benchmarks (e.g., tau-bench, tau2-bench, AppWorld) rely on fully deterministic backends, which are costly to build and iterate. We propose Proxy State-Based Evaluation, an LLM-driven simulation framework that preserves final state-based evaluation without a deterministic database. Specifically, a scenario specifies the user goal, user/system facts, expected final state, and expected agent behavior, and an LLM state tracker infers a structured proxy state from the full interaction trace. LLM judges then verify goal completion and detect tool/user hallucinations against scenario constraints. Empirically, our benchmark produces stable, model-differentiating rankings across families and inference-time reasoning efforts, and its on-/off-policy rollouts provide supervision that transfers to unseen scenarios. Careful scenario specification yields near-zero simulator hallucination rates as supported by ablation studies. The framework also supports sensitivity analyses over user personas. Human-LLM judge agreement exceeds 90%, indicating reliable automated evaluation. Overall, proxy state-based evaluation offers a practical, scalable alternative to deterministic agentic benchmarks for industrial LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19680v2">PolicyPad: Collaborative Prototyping of LLM Policies</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 CHI 2026 paper. Supplementary materials: https://docs.google.com/document/d/1jBmKXusoWmCHfwpmNhSTJtbwZ5fwVWLNppKeqqd_-pY/edit?usp=sharing
    </div>
    <details class="paper-abstract">
      As LLMs gain adoption in high-stakes domains like mental health, domain experts are increasingly consulted to provide input into policies governing their behavior. From an observation of 19 policymaking workshops with 9 experts over 15 weeks, we identified opportunities to better support rapid experimentation, feedback, and iteration for collaborative policy design processes. We present PolicyPad, an interactive system that facilitates the emerging practice of LLM policy prototyping by drawing from established UX prototyping practices, including heuristic evaluation and storyboarding. Using PolicyPad, policy designers can collaborate on drafting a policy in real time while independently testing policy-informed model behavior with usage scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain experts in mental health and law, finding that PolicyPad enhanced collaborative dynamics during policy design, enabled tight feedback loops, and led to novel policy contributions. Overall, our work paves expert-informed paths for advancing AI alignment and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10577v2">Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 fix typo and author names
    </div>
    <details class="paper-abstract">
      E-commerce campaign ranking models require large-scale training labels indicating which users purchased due to campaign influence. However, generating these labels is challenging because campaigns use creative, thematic language that does not directly map to product purchases. Without clear product-level attribution, supervised learning for campaign optimization remains limited. We present Campaign-2-PT-RAG, a scalable label generation framework that constructs user-campaign purchase labels by inferring which product types (PTs) each campaign promotes. The framework first interprets campaign content using large language models (LLMs) to capture implicit intent, then retrieves candidate PTs through semantic search over the platform taxonomy. A structured LLM-based classifier evaluates each PT's relevance, producing a campaign-specific product coverage set. User purchases matching these PTs generate positive training labels for downstream ranking models. This approach reframes the ambiguous attribution problem into a tractable semantic alignment task, enabling scalable and consistent supervision for downstream tasks such as campaign ranking optimization in production e-commerce environments. Experiments on internal and synthetic datasets, validated against expert-annotated campaign-PT mappings, show that our LLM-assisted approach generates high-quality labels with 78-90% precision while maintaining over 99% recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04072v2">Toward Beginner-Friendly LLMs for Language Learning: Controlling Difficulty in Conversation</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 EACL 2026
    </div>
    <details class="paper-abstract">
      Practicing conversations with large language models (LLMs) presents a promising alternative to traditional in-person language learning. However, most LLMs generate text at a near-native level of complexity, making them ill-suited for first and second-year beginner learners (CEFR: A1-A2). In this paper, we investigate whether controllable generation techniques can adapt LLM outputs to better support beginners. We evaluate these methods through both automatic metrics and a user study with university-level learners of Japanese. Our findings show that while prompting alone fails, controllable generation techniques can successfully improve output comprehensibility for beginner speakers (from 39.4% to 83.3%). We further introduce a new token-level evaluation metric, Token Miss Rate (TMR), that quantifies the proportion of incomprehensible tokens per utterance and correlates strongly with human judgments. To support future research in AI-assisted language learning, we release our code, models, annotation tools, and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16752v1">The Vulnerability of LLM Rankers to Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful re-rankers. Recent research has however showed that simple prompt injections embedded within a candidate document (i.e., jailbreak prompt attacks) can significantly alter an LLM's ranking decisions. While this poses serious security risks to LLM-based ranking pipelines, the extent to which this vulnerability persists across diverse LLM families, architectures, and settings remains largely under-explored. In this paper, we present a comprehensive empirical study of jailbreak prompt attacks against LLM rankers. We focus our evaluation on two complementary tasks: (1) Preference Vulnerability Assessment, measuring intrinsic susceptibility via attack success rate (ASR); and (2) Ranking Vulnerability Assessment, quantifying the operational impact on the ranking's quality (nDCG@10). We systematically examine three prevalent ranking paradigms (pairwise, listwise, setwise) under two injection variants: decision objective hijacking and decision criteria hijacking. Beyond reproducing prior findings, we expand the analysis to cover vulnerability scaling across model families, position sensitivity, backbone architectures, and cross-domain robustness. Our results characterize the boundary conditions of these vulnerabilities, revealing critical insights such as that encoder-decoder architectures exhibit strong inherent resilience to jailbreak attacks. We publicly release our code and additional experimental results at https://github.com/ielab/LLM-Ranker-Attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16534v5">Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore
    </div>
    <details class="paper-abstract">
      Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks. The code is available at https://github.com/jcnf0/targeting-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16162v1">LLMs Exhibit Significantly Lower Uncertainty in Creative Writing Than Professional Writers</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 11 tables
    </div>
    <details class="paper-abstract">
      We argue that uncertainty is a key and understudied limitation of LLMs' performance in creative writing, which is often characterized as trite and cliché-ridden. Literary theory identifies uncertainty as a necessary condition for creative expression, while current alignment strategies steer models away from uncertain outputs to ensure factuality and reduce hallucination. We formalize this tension by quantifying the "uncertainty gap" between human-authored stories and model-generated continuations. Through a controlled information-theoretic analysis of 28 LLMs on high-quality storytelling datasets, we demonstrate that human writing consistently exhibits significantly higher uncertainty than model outputs. We find that instruction-tuned and reasoning models exacerbate this trend compared to their base counterparts; furthermore, the gap is more pronounced in creative writing than in functional domains, and strongly correlates to writing quality. Achieving human-level creativity requires new uncertainty-aware alignment paradigms that can distinguish between destructive hallucinations and the constructive ambiguity required for literary richness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03310v3">Randomized Masked Finetuning: An Efficient Way to Mitigate Memorization of PIIs in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      The current literature on memorization in Natural Language Models, especially Large Language Models (LLMs), poses severe security and privacy risks, as models tend to memorize personally identifying information (PIIs) from training data. We introduce Randomized Masked Fine-Tuning (RMFT), a novel privacy-preserving fine-tuning technique that reduces PII memorization while minimizing performance impact. Using the Enron Email Dataset, we demonstrate that RMFT achieves an 80.81% reduction in Total Extraction Rate and 80.17% reduction in Seen Extraction Rate compared to baseline fine-tuning, outperforming deduplication methods while maintaining only a 5.73% increase in perplexity. We present MaxTER, a Pareto-optimal evaluation framework for assessing privacy-utility tradeoffs, and show the performance of RMFT vs Deduplication by Area Under The Response Curve (AURC) metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16131v1">Empirical Cumulative Distribution Function Clustering for LLM-based Agent System Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as agents to solve complex tasks such as question answering (QA), scientific debate, and software development. A standard evaluation procedure aggregates multiple responses from LLM agents into a single final answer, often via majority voting, and compares it against reference answers. However, this process can obscure the quality and distributional characteristics of the original responses. In this paper, we propose a novel evaluation framework based on the empirical cumulative distribution function (ECDF) of cosine similarities between generated responses and reference answers. This enables a more nuanced assessment of response quality beyond exact match metrics. To analyze the response distributions across different agent configurations, we further introduce a clustering method for ECDFs using their distances and the $k$-medoids algorithm. Our experiments on a QA dataset demonstrate that ECDFs can distinguish between agent settings with similar final accuracies but different quality distributions. The clustering analysis also reveals interpretable group structures in the responses, offering insights into the impact of temperature, persona, and question topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11348v2">AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16106v1">Algorithm-Based Pipeline for Reliable and Intent-Preserving Code Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 Accepted at 34th IEEE/ACM International Conference on Program Comprehension (ICPC 2026)
    </div>
    <details class="paper-abstract">
      Code translation, the automatic conversion of programs between languages, is a growing use case for Large Language Models (LLMs). However, direct one-shot translation often fails to preserve program intent, leading to errors in control flow, type handling, and I/O behavior. We propose an algorithm-based pipeline that introduces a language-neutral intermediate specification to capture these details before code generation. This study empirically evaluates the extent to which structured planning can improve translation accuracy and reliability relative to direct translation. We conduct an automated paired experiment - direct and algorithm-based to translate between Python and Java using five widely used LLMs on the Avatar and CodeNet datasets. For each combination (model, dataset, approach, and direction), we compile and execute the translated program and run the tests provided. We record compilation results, runtime behavior, timeouts (e.g., infinite loop), and test outcomes. We compute accuracy from these tests, counting a translation as correct only if it compiles, runs without exceptions or timeouts, and passes all tests. We then map every failed compile-time and runtime case to a unified, language-aware taxonomy and compare subtype frequencies between the direct and algorithm-based approaches. Overall, the Algorithm-based approach increases micro-average accuracy from 67.7% to 78.5% (10.8% increase). It eliminates lexical and token errors by 100%, reduces incomplete constructs by 72.7%, and structural and declaration issues by 61.1%. It also substantially lowers runtime dependency and entry-point failures by 78.4%. These results demonstrate that algorithm-based pipelines enable more reliable, intent-preserving code translation, providing a foundation for robust multilingual programming assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16741v1">Can Adversarial Code Comments Fool AI Security Reviewers -- Large-Scale Empirical Study of Comment-Based Attacks and Defenses Against LLM Code Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 19 pages, 6 figures
    </div>
    <details class="paper-abstract">
      AI-assisted code review is widely used to detect vulnerabilities before production release. Prior work shows that adversarial prompt manipulation can degrade large language model (LLM) performance in code generation. We test whether similar comment-based manipulation misleads LLMs during vulnerability detection. We build a 100-sample benchmark across Python, JavaScript, and Java, each paired with eight comment variants ranging from no comments to adversarial strategies such as authority spoofing and technical deception. Eight frontier models, five commercial and three open-source, are evaluated in 9,366 trials. Adversarial comments produce small, statistically non-significant effects on detection accuracy (McNemar exact p > 0.21; all 95 percent confidence intervals include zero). This holds for commercial models with 89 to 96 percent baseline detection and open-source models with 53 to 72 percent, despite large absolute performance gaps. Unlike generation settings where comment manipulation achieves high attack success, detection performance does not meaningfully degrade. More complex adversarial strategies offer no advantage over simple manipulative comments. We test four automated defenses across 4,646 additional trials (14,012 total). Static analysis cross-referencing performs best at 96.9 percent detection and recovers 47 percent of baseline misses. Comment stripping reduces detection for weaker models by removing helpful context. Failures concentrate on inherently difficult vulnerability classes, including race conditions, timing side channels, and complex authorization logic, rather than on adversarial comments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16100v1">LLM-Driven Intent-Based Privacy-Aware Orchestration Across the Cloud-Edge Continuum</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), efficiently serving LLM inference under limited GPU resources has become a critical challenge. Recently, an increasing number of studies have explored applying serverless computing paradigms to LLM serving in order to maximize resource utilization. However, LLM inference workloads are highly diverse, and modern GPU clusters are inherently heterogeneous, making it necessary to dynamically adjust deployment configurations online to better adapt to the elastic and dynamic nature of serverless environments. At the same time, enabling such online reconfiguration is particularly challenging due to the stateful nature of LLM inference and the massive size of model parameters. In this paper, we propose a dynamic pipeline reconfiguration approach that enables online adjustment of pipeline configurations while minimizing service downtime and performance degradation. Our method allows the system to select the optimal pipeline configuration in response to changing workloads. Experimental results on heterogeneous GPU platforms, including NVIDIA A100 and L40s, demonstrate that our migration mechanism incurs less than 50 ms of service downtime, while introducing under 10% overhead on both time-to-first-token (TTFT) and time-per-output-token (TPOT).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25380v2">Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Data curriculums have become central to successful LLM training, yet principles governing optimal data placement remain unclear. We introduce the *training re-evaluation curve (TREC)*, a diagnostic that retrospectively evaluates training batches *using the final model weights*. The TREC characterizes how well a trained model retains training data as a function of *when* the data was encountered during training. Analyzing TRECs for models from 111M to 3.9B parameters, we show that placing high-quality data at low points on the TREC significantly improves performance. Importantly, while a TREC is initially observable only after training, we demonstrate it can be *predicted in advance* from AdamW's implicit EMA coefficients, enabling proactive curriculum design. By predicting TRECs for published training recipes, we explain prior ablations and reveal suboptimal data placements. We also align high-quality data with TREC minima in order to improve continual pre-training of a 3.9B-parameter LLM trained on 900B tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00364v3">"Someone Hid It": Query-Agnostic Black-Box Attacks on LLM-Based Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been serving as effective backbones for retrieval systems, including Retrieval-Augmentation-Generation (RAG), Dense Information Retriever (IR), and Agent Memory Retrieval. Recent studies have demonstrated that such LLM-based Retrieval (LLMR) is vulnerable to adversarial attacks, which manipulates documents by token-level injections and enables adversaries to either boost or diminish these documents in retrieval tasks. However, existing attack studies mainly (1) presume a known query is given to the attacker, and (2) highly rely on access to the victim model's parameters or interactions, which are hardly accessible in real-world scenarios, leading to limited validity. To further explore the secure risks of LLMR, we propose a practical black-box attack method that generates transferable injection tokens based on zero-shot surrogate LLMs without need of victim queries or victim models knowledge. The effectiveness of our attack raises such a robustness issue that similar effects may arise from benign or unintended document edits in the real world. To achieve our attack, we first establish a theoretical framework of LLMR and empirically verify it. Under the framework, we simulate the transferable attack as a min-max problem, and propose an adversarial learning mechanism that finds optimal adversarial tokens with learnable query samples. Our attack is validated to be effective on benchmark datasets across popular LLM retrievers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16054v1">CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The prefill stage in long-context LLM inference remains a computational bottleneck. Recent token-ranking heuristics accelerate inference by selectively processing a subset of semantically relevant tokens. However, existing methods suffer from unstable token importance estimation, often varying between layers. Evaluating token-ranking quality independently from heuristic-specific architectures is challenging. To address this, we introduce an Answer-Informed Oracle, which defines ground-truth token importance by measuring attention from generated answers back to the prompt. This oracle reveals that existing heuristics exhibit high variance across layers: rankings can degrade sharply at specific layers, a failure mode invisible to end-to-end benchmarks. The diagnosis suggests a simple fix: aggregate scores across layers rather than relying on any single one. We implement this as Cross-Layer Attention Aggregation (CLAA), which closes the gap to the oracle upper bound and reduces Time-to-First-Token (TTFT) by up to 39\% compared to the Full KV Cache baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16039v1">How Uncertain Is the Grade? A Benchmark of Uncertainty Metrics for LLM-Based Automatic Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) is reshaping the landscape of automatic assessment in education. While these systems demonstrate substantial advantages in adaptability to diverse question types and flexibility in output formats, they also introduce new challenges related to output uncertainty, stemming from the inherently probabilistic nature of LLMs. Output uncertainty is an inescapable challenge in automatic assessment, as assessment results often play a critical role in informing subsequent pedagogical actions, such as providing feedback to students or guiding instructional decisions. Unreliable or poorly calibrated uncertainty estimates can lead to unstable downstream interventions, potentially disrupting students' learning processes and resulting in unintended negative consequences. To systematically understand this challenge and inform future research, we benchmark a broad range of uncertainty quantification methods in the context of LLM-based automatic assessment. Although the effectiveness of these methods has been demonstrated in many tasks across other domains, their applicability and reliability in educational settings, particularly for automatic grading, remain underexplored. Through comprehensive analyses of uncertainty behaviors across multiple assessment datasets, LLM families, and generation control settings, we characterize the uncertainty patterns exhibited by LLMs in grading scenarios. Based on these findings, we evaluate the strengths and limitations of different uncertainty metrics and analyze the influence of key factors, including model families, assessment tasks, and decoding strategies, on uncertainty estimates. Our study provides actionable insights into the characteristics of uncertainty in LLM-based automatic assessment and lays the groundwork for developing more reliable and effective uncertainty-aware grading systems in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16034v1">FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 Accepted to The Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      Federated cross-domain recommendation (Federated CDR) aims to collaboratively learn personalized recommendation models across heterogeneous domains while preserving data privacy. Recently, large language model (LLM)-based recommendation models have demonstrated impressive performance by leveraging LLMs' strong reasoning capabilities and broad knowledge. However, adopting LLM-based recommendation models in Federated CDR scenarios introduces new challenges. First, there exists a risk of overfitting with domain-specific local adapters. The magnitudes of locally optimized parameter updates often vary across domains, causing biased aggregation and overfitting toward domain-specific distributions. Second, unlike traditional recommendation models (e.g., collaborative filtering, bipartite graph-based methods) that learn explicit and comparable user/item representations, LLMs encode knowledge implicitly through autoregressive text generation training. This poses additional challenges for effectively measuring the cross-domain similarities under heterogeneity. To address these challenges, we propose an LLM-based framework for federated cross-domain recommendation, FeDecider. Specifically, FeDecider tackles the challenge of scale-specific noise by disentangling each client's low-rank updates and sharing only their directional components. To handle the need for flexible and effective integration, each client further learns personalized weights that achieve the data-aware integration of updates from other domains. Extensive experiments across diverse datasets validate the effectiveness of our proposed FeDecider.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16735v1">A Few-Shot LLM Framework for Extreme Day Classification in Electricity Markets</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      This paper proposes a few-shot classification framework based on Large Language Models (LLMs) to predict whether the next day will have spikes in real-time electricity prices. The approach aggregates system state information, including electricity demand, renewable generation, weather forecasts, and recent electricity prices, into a set of statistical features that are formatted as natural-language prompts and fed to an LLM along with general instructions. The model then determines the likelihood that the next day would be a spike day and reports a confidence score. Using historical data from the Texas electricity market, we demonstrate that this few-shot approach achieves performance comparable to supervised machine learning models, such as Support Vector Machines and XGBoost, and outperforms the latter two when limited historical data are available. These findings highlight the potential of LLMs as a data-efficient tool for classifying electricity price spikes in settings with scarce data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15983v1">ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 Code and benchmark: \url{https://github.com/junbolian/ReLoop}
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can translate natural language into optimization code, but silent failures pose a critical risk: code that executes and returns solver-feasible solutions may encode semantically incorrect formulations, creating a feasibility-correctness gap of up to 90 percentage points on compositional problems. We introduce ReLoop, addressing silent failures from two complementary directions. Structured generation decomposes code production into a four-stage reasoning chain (understand, formalize, synthesize, verify) that mirrors expert modeling practice, with explicit variable-type reasoning and self-verification to prevent formulation errors at their source. Behavioral verification detects errors that survive generation by testing whether the formulation responds correctly to solver-based parameter perturbation, without requiring ground truth -- an external semantic signal that bypasses the self-consistency problem inherent in LLM-based code review. The two mechanisms are complementary: structured generation dominates on complex compositional problems, while behavioral verification becomes the largest single contributor on problems with localized formulation defects. Together with execution recovery via IIS-enhanced diagnostics, ReLoop raises correctness from 22.6% to 31.1% and execution from 72.1% to 100.0% on the strongest model, with consistent gains across five models spanning three paradigms (foundation, SFT, RL) and three benchmarks. We additionally release RetailOpt-190, 190 compositional retail optimization scenarios targeting the multi-constraint interactions where LLMs most frequently fail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15823v1">CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      A central challenge in large language model (LLM) editing is capability preservation: methods that successfully change targeted behavior can quietly game the editing proxy and corrupt general capabilities, producing degenerate behaviors reminiscent of proxy/reward hacking. We present CrispEdit, a scalable and principled second-order editing algorithm that treats capability preservation as an explicit constraint, unifying and generalizing several existing editing approaches. CrispEdit formulates editing as constrained optimization and enforces the constraint by projecting edit updates onto the low-curvature subspace of the capability-loss landscape. At the crux of CrispEdit is expressing capability constraint via Bregman divergence, whose quadratic form yields the Gauss-Newton Hessian exactly and even when the base model is not trained to convergence. We make this second-order procedure efficient at the LLM scale using Kronecker-factored approximate curvature (K-FAC) and a novel matrix-free projector that exploits Kronecker structure to avoid constructing massive projection matrices. Across standard model-editing benchmarks, CrispEdit achieves high edit success while keeping capability degradation below 1% on average across datasets, significantly improving over prior editors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15785v1">This human study did not involve human subjects: Validating LLM simulations as behavioral evidence</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      A growing literature uses large language models (LLMs) as synthetic participants to generate cost-effective and nearly instantaneous responses in social science experiments. However, there is limited guidance on when such simulations support valid inference about human behavior. We contrast two strategies for obtaining valid estimates of causal effects and clarify the assumptions under which each is suitable for exploratory versus confirmatory research. Heuristic approaches seek to establish that simulated and observed human behavior are interchangeable through prompt engineering, model fine-tuning, and other repair strategies designed to reduce LLM-induced inaccuracies. While useful for many exploratory tasks, heuristic approaches lack the formal statistical guarantees typically required for confirmatory research. In contrast, statistical calibration combines auxiliary human data with statistical adjustments to account for discrepancies between observed and simulated responses. Under explicit assumptions, statistical calibration preserves validity and provides more precise estimates of causal effects at lower cost than experiments that rely solely on human participants. Yet the potential of both approaches depends on how well LLMs approximate the relevant populations. We consider what opportunities are overlooked when researchers focus myopically on substituting LLMs for human participants in a study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15778v1">*-PLUIE: Personalisable metric with Llm Used for Improved Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Evaluating the quality of automatically generated text often relies on LLM-as-a-judge (LLM-judge) methods. While effective, these approaches are computationally expensive and require post-processing. To address these limitations, we build upon ParaPLUIE, a perplexity-based LLM-judge metric that estimates confidence over ``Yes/No'' answers without generating text. We introduce *-PLUIE, task specific prompting variants of ParaPLUIE and evaluate their alignment with human judgement. Our experiments show that personalised *-PLUIE achieves stronger correlations with human ratings while maintaining low computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15761v1">A Differential Fuzzing-Based Evaluation of Functional Equivalence in LLM-Generated Code Refactorings</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      With the rapid adoption of large language models (LLMs) in automated code refactoring, assessing and ensuring functional equivalence between LLM-generated refactoring and the original implementation becomes critical. While prior work typically relies on predefined test cases to evaluate correctness, in this work, we leverage differential fuzzing to check functional equivalence in LLM-generated code refactorings. Unlike test-based evaluation, a differential fuzzing-based equivalence checker needs no predefined test cases and can explore a much larger input space by executing and comparing thousands of automatically generated test inputs. In a large-scale evaluation of six LLMs (CodeLlama, Codestral, StarChat2, Qwen-2.5, Olmo-3, and GPT-4o) across three datasets and two refactoring types, we find that LLMs show a non-trivial tendency to alter program semantics, producing 19-35% functionally non-equivalent refactorings. Our experiments further demonstrate that about 21% of these non-equivalent refactorings remain undetected by the existing test suites of the three evaluated datasets. Collectively, the findings of this study imply that reliance on existing tests might overestimate functional equivalence in LLM-generated code refactorings, which remain prone to semantic divergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15753v1">Under-resourced studies of under-resourced languages: lemmatization and POS-tagging with LLM annotators for historical Armenian, Georgian, Greek and Syriac</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Low-resource languages pose persistent challenges for Natural Language Processing tasks such as lemmatization and part-of-speech (POS) tagging. This paper investigates the capacity of recent large language models (LLMs), including GPT-4 variants and open-weight Mistral models, to address these tasks in few-shot and zero-shot settings for four historically and linguistically diverse under-resourced languages: Ancient Greek, Classical Armenian, Old Georgian, and Syriac. Using a novel benchmark comprising aligned training and out-of-domain test corpora, we evaluate the performance of foundation models across lemmatization and POS-tagging, and compare them with PIE, a task-specific RNN baseline. Our results demonstrate that LLMs, even without fine-tuning, achieve competitive or superior performance in POS-tagging and lemmatization across most languages in few-shot settings. Significant challenges persist for languages characterized by complex morphology and non-Latin scripts, but we demonstrate that LLMs are a credible and relevant option for initiating linguistic annotation tasks in the absence of data, serving as an effective aid for annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10134v2">FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Uncrewed Aerial Vehicles (UAVs) play a vital role in public safety, especially in monitoring wildfires, where early detection reduces environmental impact. In UAV-Assisted Wildfire Monitoring (UAWM) systems, jointly optimizing the data collection schedule and UAV velocity is essential to minimize the average Age of Information (AoI) for sensory data. Deep Reinforcement Learning (DRL) has been used for this optimization, but its limitations-including low sampling efficiency, discrepancies between simulation and real-world conditions, and complex training make it unsuitable for time-critical applications such as wildfire monitoring. Recent advances in Large Language Models (LLMs) provide a promising alternative. With strong reasoning and generalization capabilities, LLMs can adapt to new tasks through In-Context Learning (ICL), which enables task adaptation using natural language prompts and example-based guidance without retraining. This paper proposes a novel online Flight Resource Allocation scheme based on LLM-Enabled In-Context Learning (FRSICL) to jointly optimize the data collection schedule and UAV velocity along the trajectory in real time, thereby asymptotically minimizing the average AoI across all ground sensors. Unlike DRL, FRSICL generates data collection schedules and velocities using natural language task descriptions and feedback from the environment, enabling dynamic decision-making without extensive retraining. Simulation results confirm the effectiveness of FRSICL compared to state-of-the-art baselines, namely Proximal Policy Optimization, Block Coordinate Descent, and Nearest Neighbor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17812v2">Can Multimodal LLMs Perform Time Series Anomaly Detection?</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 ACM Web Conference 2026 (WWW'26)
    </div>
    <details class="paper-abstract">
      Time series anomaly detection (TSAD) has been a long-standing pillar problem in Web-scale systems and online infrastructures, such as service reliability monitoring, system fault diagnosis, and performance optimization. Large language models (LLMs) have demonstrated unprecedented capabilities in time series analysis, the potential of multimodal LLMs (MLLMs), particularly vision-language models, in TSAD remains largely under-explored. One natural way for humans to detect time series anomalies is through visualization and textual description. It motivates our research question: Can multimodal LLMs perform time series anomaly detection? Existing studies often oversimplify the problem by treating point-wise anomalies as special cases of range-wise ones or by aggregating point anomalies to approximate range-wise scenarios. They limit our understanding for realistic scenarios such as multi-granular anomalies and irregular time series. To address the gap, we build a VisualTimeAnomaly benchmark to comprehensively investigate zero-shot capabilities of MLLMs for TSAD, progressively from point-, range-, to variate-wise anomalies, and extends to irregular sampling conditions. Our study reveals several key insights in multimodal MLLMs for TSAD. Built on these findings, we propose a MLLMs-based multi-agent framework TSAD-Agents to achieve automatic TSAD. Our framework comprises scanning, planning, detection, and checking agents that synergistically collaborate to reason, plan, and self-reflect to enable automatic TSAD. These agents adaptively invoke tools such as traditional methods and MLLMs and dynamically switch between text and image modalities to optimize detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15681v1">A universal LLM Framework for General Query Refinements</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Numerous studies have explored the SQL query refinement problem, where the objective is to minimally modify an input query so that it satisfies a specified set of constraints. However, these works typically target restricted classes of queries or constraints. We present OmniTune, a general framework for refining arbitrary SQL queries using LLM-based optimization by prompting (OPRO). OmniTune employs a two-step OPRO scheme that explores promising refinement subspaces and samples candidates within them, supported by concise history and skyline summaries for effective feedback. Experiments on a comprehensive benchmark demonstrate that OmniTune handles both previously studied refinement tasks and more complex scenarios beyond the scope of existing solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15675v1">LLM-to-Speech: A Synthetic Data Pipeline for Training Dialectal Text-to-Speech Models</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 8 pages, 2 figures, EACL26
    </div>
    <details class="paper-abstract">
      Despite the advances in neural text to speech (TTS), many Arabic dialectal varieties remain marginally addressed, with most resources concentrated on Modern Spoken Arabic (MSA) and Gulf dialects, leaving Egyptian Arabic -- the most widely understood Arabic dialect -- severely under-resourced. We address this gap by introducing NileTTS: 38 hours of transcribed speech from two speakers across diverse domains including medical, sales, and general conversations. We construct this dataset using a novel synthetic pipeline: large language models (LLM) generate Egyptian Arabic content, which is then converted to natural speech using audio synthesis tools, followed by automatic transcription and speaker diarization with manual quality verification. We fine-tune XTTS v2, a state-of-the-art multilingual TTS model, on our dataset and evaluate against the baseline model trained on other Arabic dialects. Our contributions include: (1) the first publicly available Egyptian Arabic TTS dataset, (2) a reproducible synthetic data generation pipeline for dialectal TTS, and (3) an open-source fine-tuned model. All resources are released to advance Egyptian Arabic speech synthesis research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16727v1">Mobility-Aware Cache Framework for Scalable LLM-Based Human Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Large-scale human mobility simulation is critical for applications such as urban planning, epidemiology, and transportation analysis. Recent works treat large language models (LLMs) as human agents to simulate realistic mobility behaviors using structured reasoning, but their high computational cost limits scalability. To address this, we design a mobility-aware cache framework named MobCache that leverages reconstructible caches to enable efficient large-scale human mobility simulations. It consists of: (1) a reasoning component that encodes each reasoning step as a latent-space embedding and uses a latent-space evaluator to enable the reuse and recombination of reasoning steps; and (2) a decoding component that employs a lightweight decoder trained with mobility law-constrained distillation to translate latent-space reasoning chains into natural language, thereby improving simulation efficiency while maintaining fidelity. Experiments show that MobCache significantly improves efficiency across multiple dimensions while maintaining performance comparable to state-of-the-art LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16726v1">Guiding LLM-Based Human Mobility Simulation with Mobility Measures from Shared Data</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Large-scale human mobility simulation is critical for many science domains such as urban science, epidemiology, and transportation analysis. Recent works treat large language models (LLMs) as human agents to simulate realistic mobility trajectories by modeling individual-level cognitive processes. However, these approaches generate individual mobility trajectories independently, without any population-level coordination mechanism, and thus fail to capture the emergence of collective behaviors. To address this issue, we design M2LSimu, a mobility measures-guided multi-prompt adjustment framework that leverages mobility measures derived from shared data as guidance to refine individual-level prompts for realistic mobility generation. Our framework applies coarse-grained adjustment strategies guided by mobility measures, progressively enabling fine-grained individual-level adaptation while satisfying multiple population-level mobility objectives under a limited budget. Experiments show that M2LSimu significantly outperforms state-of-the-art LLM-based methods on two public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15654v1">Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Self-evolving LLM agents update their internal state across sessions, often by writing and reusing long-term memory. This design improves performance on long-horizon tasks but creates a security risk: untrusted external content observed during a benign session can be stored as memory and later treated as instruction. We study this risk and formalize a persistent attack we call a Zombie Agent, where an attacker covertly implants a payload that survives across sessions, effectively turning the agent into a puppet of the attacker. We present a black-box attack framework that uses only indirect exposure through attacker-controlled web content. The attack has two phases. During infection, the agent reads a poisoned source while completing a benign task and writes the payload into long-term memory through its normal update process. During trigger, the payload is retrieved or carried forward and causes unauthorized tool behavior. We design mechanism-specific persistence strategies for common memory implementations, including sliding-window and retrieval-augmented memory, to resist truncation and relevance filtering. We evaluate the attack on representative agent setups and tasks, measuring both persistence over time and the ability to induce unauthorized actions while preserving benign task quality. Our results show that memory evolution can convert one-time indirect injection into persistent compromise, which suggests that defenses focused only on per-session prompt filtering are not sufficient for self-evolving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15311v3">Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 v3: Production hardening. Added INT8 quantization (5.6x dot product speedup, 3.1x compression), crash recovery via decoupled WAL (<1% overhead), unlimited text storage via sidecar blob arena with generational GC, and epoch-based reclamation for lock-free reads (P99 750ns under 16-thread contention). Revised for systems engineering clarity
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are fundamentally constrained by the quadratic computational cost of self-attention and the "Lost in the Middle" phenomenon, where reasoning capabilities degrade as context windows expand. Existing solutions, primarily "Flat RAG" architectures relying on vector databases, treat memory as an unstructured bag of embeddings, failing to capture the hierarchical and temporal structure of long-horizon interactions. This paper presents Aeon, a Neuro-Symbolic Cognitive Operating System that redefines memory as a managed OS resource. Aeon structures memory into a Memory Palace (a spatial index implemented via Atlas, a SIMD-accelerated Page-Clustered Vector Index) and a Trace (a neuro-symbolic episodic graph). This architecture introduces three advances: (1) Symmetric INT8 Scalar Quantization, achieving 3.1x spatial compression and 5.6x math acceleration via NEON SDOT intrinsics; (2) a decoupled Write-Ahead Log (WAL) ensuring crash-recoverability with statistically negligible overhead (<1%); and (3) a Sidecar Blob Arena eliminating the prior 440-character text ceiling via an append-only mmap-backed blob file with generational garbage collection. The Semantic Lookaside Buffer (SLB) exploits conversational locality to achieve sub-5us retrieval latencies, with INT8 vectors dequantized to FP32 on cache insertion to preserve L1-resident lookup performance. Benchmarks on Apple M4 Max demonstrate that the combined architecture achieves 4.70ns INT8 dot product latency, 3.09us tree traversal at 100K nodes (3.4x over FP32), and P99 read latency of 750ns under hostile 16-thread contention via epoch-based reclamation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03581v3">Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities and highlighting the potential for safer and more collaborative agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07812v2">LLMs Know More About Numbers than They Can Say</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 EACL 2026 (Oral), camera-ready version with GitHub link
    </div>
    <details class="paper-abstract">
      Although state-of-the-art LLMs can solve math problems, we find that they make errors on numerical comparisons with mixed notation: "Which is larger, $5.7 \times 10^2$ or $580$?" This raises a fundamental question: Do LLMs even know how big these numbers are? We probe the hidden states of several smaller open-source LLMs. A single linear projection of an appropriate hidden layer encodes the log-magnitudes of both kinds of numerals, allowing us to recover the numbers with relative error of about 2.3% (on restricted synthetic text) or 19.06% (on scientific papers). Furthermore, the hidden state after reading a pair of numerals encodes their ranking, with a linear classifier achieving over 90% accuracy. Yet surprisingly, when explicitly asked to rank the same pairs of numerals, these LLMs achieve only 50-70% accuracy, with worse performance for models whose probes are less effective. Finally, we show that incorporating the classifier probe's log-loss as an auxiliary objective during finetuning brings an additional 3.22% improvement in verbalized accuracy over base models, demonstrating that improving models' internal magnitude representations can enhance their numerical reasoning capabilities. Our code is available at https://github.com/VCY019/Numeracy-Probing.
    </details>
</div>
