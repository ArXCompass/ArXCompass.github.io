# llm - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20573v1">Beyond LLM Judge: Mechanical Verification for Agentic Autoformalization</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      We introduce a novel method for verifying Lean formalizations that goes beyond LLM-judged faithfulness. Unlike most verification methods, which rely on LLMs or trained models, our method combines LLM-based semantic review with mechanical numerical and vacuity checks, providing checkable feedback and detecting errors missed by LLM judges. We apply this method, along with our Extract-Verify-Proof autoformalization pipeline, to construct the Formalized Library of Mathematical Functions (FLMF), a partial formalization of the NIST Digital Library of Mathematical Functions (DLMF). On three labeled datasets, our method achieves the lowest accepted-error rate among the evaluated methods, measured as the fraction of incorrect statements accepted by the verification method. Expert review of the results also identifies translation and source defects in existing datasets. Code available at https://github.com/annamei1/FLMF-Ch7ErrFunc-Release.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20570v1">Manacá-1B-Instruct-2: Reproducible Safety Alignment for a Native Brazilian-Portuguese LLM. Negative DPO Results, a Controlled Composition Scan, and a Silent Failure Mode of the LLM Judge</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Brazilian Portuguese remains underserved by open-source language models, and the few that are safety-aligned rarely document what did not work or validate comparisons with statistical significance. We present Manacá-1B-Instruct-2, the second-generation post-training alignment of Manacá-1B (1.72B parameters, trained from scratch for PT-BR), with an SFT corpus of 291,223 examples. We document three DPO attempts, all negative, including a case where the training reward margin reaches 1.0 while free generation does not change at all, and show that safety came from a dedicated safety-SFT (LLM-jp AnswerCarefully style), refined over three iterations until closing the safety-helpfulness frontier. A controlled scan of the fraction of general data in the safety-SFT mix (10% to 60%) reveals a non-monotonic over-refusal curve with a knee point at 40-50%. The final model is compared against four external PT-BR models on the same items, with Wilson confidence intervals and paired McNemar tests with Holm-Bonferroni correction: the lower bound of its safety interval (94.1%) lies above the upper bound of all four baselines. We also report a methodological finding for LLM-judge evaluation: on two of six safety items in MT-Bench-PT, the judge categorically refuses to assign a score regardless of the evaluated response, silently reducing the effective N of that category in every prior report of the project. We complement this with an n-gram decontamination check and a Chinchilla-anchored compute-budget table that attributes the gap in general capability to pre-training scale rather than alignment methodology. All code, raw data, logs, and scripts that produce every reported number are publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02022v2">Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      AI-based ideation systems are often evaluated on the novelty of their ideas, and that judgment is increasingly delegated to large language models. Such judges are typically built ad hoc and validated, if at all, on human-authored papers rather than on the generated ideas they are meant to score. So, how do novelty judges perform? Not well. We present a systematic controlled study of novelty evaluation design choices. We build an evaluation set automatically, mining OpenReview for passages where reviewers explicitly affirm or dispute a paper's originality and keeping only submissions with unanimous agreement at the extremes of their research area; we pair these with ideas from a vanilla LLM generator. Across six judges, we find that seemingly small prompt design choices have large consequences; e.g., simply telling the judge that reviewers found one idea novel and the other not can change its verdict on more than half of the identical idea pairs it is shown, shifting pairwise accuracy by over 50 points and occasionally pushing it below chance. The same change helps one judge and hurts another. Retrieval and larger reasoning budgets help little, and two purpose-built novelty evaluators are outperformed by our cheapest prompted baseline. These results raise questions about reported novelty gains of automated ideation systems, and call for robust novelty evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20548v1">PerSo: Investigating Facets of LLMs' Social Intelligence in Real-World Persuasion Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 25 pages, 17 figures, 10 tables. Ali Zamani and Soroush Esfahanian contributed equally
    </div>
    <details class="paper-abstract">
      Understanding the social intelligence of large language models (LLMs) is increasingly important for applications involving human-AI interaction and multi-agent environments. Social intelligence comprises two aspects: cognitive intelligence, the ability to infer others' mental states, and behavioral intelligence, the ability to select appropriate social actions based on these inferences to achieve social goals. Persuasion provides a natural testbed in which both aspects are simultaneously at play, yet prior benchmarks typically evaluate them in isolation using synthetic scenarios. We introduce PerSo, a benchmark that evaluates both aspects on real-world persuasion dialogues from Reddit's ChangeMyView community, using an LLM-in-the-loop pipeline to construct the evaluation tasks, validated through human annotation. Our evaluation of diverse LLMs shows that performance generally scales with model size but varies across dimensions. The strongest LLM slightly surpasses the best observed human performance on the behavioral task, while the best human participant remains ahead on the cognitive task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17193v3">Multi-LLM Systems Exhibit Robust Semantic Collapse</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 94 pages, 12 figures, 13 tables; includes Supplementary Information
    </div>
    <details class="paper-abstract">
      Whether machines can originate novel content has been debated for nearly two centuries, from Lovelace's assertion that no engine can "originate anything" to Turing's question of whether a machine can amplify ideas brought in from outside. Interacting multi-LLM systems, increasingly deployed for autonomous generation, reopen this question empirically. Here we show that such systems exhibit semantic collapse, systematic convergence in semantic representations despite apparent lexical variation, with no retraining or weight updates. The pattern holds across model families and over 200 to 1,000 rounds. Thirteen intervention implementations tested, spanning decoding parameters, prompt design, agent composition, activation engineering, and reinforcement learning, did not reliably restore semantic diversity. Mechanistic analyses suggest that semantic collapse is not solely attributable to the alignment or conformity biases examined here, but is consistent with intrinsic properties of autoregressive generation. Our results point to persistent constraints on the ability of multi-LLM systems to sustain open-ended exploration in closed-loop settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26277v3">From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Dianjing Fan and Yao Li equally contributed to this work. 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach transforms raw web clickstreams into two outputs: a self-supervised Transformer encodes multi-modal clickstreams into a compact session embedding, while an LLM-based taxonomy generation and distillation pipeline produces interpretable intent labels. Our system demonstrates that self-supervised clickstream representations combined with LLM-distilled taxonomies can jointly serve quantitative tasks and qualitative understanding in production: on the mobile homepage tile ranking task, the session embedding improves macro Recall@1 by 1.88% and reduces Log Loss by 13.38% over production baselines. On the user conversion prediction task, the embedding outperforms the LLM labels by 4.3% on micro F1, while the distillation layer delivers interpretable labels at ultra-low latency with only a 7% performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09366v2">Is Text All You Need? Text as a Universal Information Bottleneck for Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide a powerful reasoning backbone for speech understanding, but integrating continuous acoustic signals into a pretrained LLM remains challenging. A central design question is how to tie speech representations to the LLM's text input space while preserving the continuous variation needed for different speech tasks. In this work, we propose Convex Gate (C-Gate), a speech-LLM bridge that represents each frame as a convex combination of frozen token embeddings. This architectural convex-hull constraint fixes the output codebook while allowing continuous mixtures learned from task supervision. With a Whisper-large-v3 encoder, a partially adapted Qwen2.5-7B-Instruct decoder, and joint training on speech recognition, emotion recognition, and spoken reasoning, C-Gate performs accurate transcription and emotion recognition, our representative semantic and acoustic tasks, through one shared decoder. On text-only benchmarks, the adapted decoder loses instruction-following and MMLU accuracy and matches or exceeds the original model on GSM8K and HellaSwag. We further design analyses and interventions on the trained model to examine where the bridge sequence carries task information and how the decoder reads it. These findings show how fixed text embeddings can support both semantic and acoustic readouts through continuous speech-conditioned mixtures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04196v2">Asynchrony Is Nearly Free for Evolution Strategies on Long-Horizon LLM Agentic Tasks</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Synchronous policy updates can limit the efficiency of post-training for long-horizon LLM agents. When trajectories span multiple interaction turns, completion times can vary substantially, and synchronous update barriers leave faster workers idling, wasting compute resources. Asynchronous reinforcement learning addresses this inefficiency but introduces policy lag and off-policy optimization. Evolution strategies (ES) are a backpropagation-free alternative to gradient-based training, yet require many perturbed-policy evaluations and have remained largely synchronous in current LLM post-training. We introduce bounded-staleness asynchronous ES and investigate its training behavior under stale fitness measurements. On WebShop, we systematically vary the lag and fraction of stale fitness measurements and additionally introduce two stress tests to probe extreme staleness cases. Natural asynchrony closely tracks synchronous ES, and degradation in evaluation accuracy remains small even under extreme staleness, including when all fitness measurements are stale. These findings suggest that ES can tolerate substantial policy staleness without explicit off-policy correction, supporting its use in asynchronous post-training for long-horizon LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20685v1">L2G-SR: Learning to Guide LLM-based Symbolic Regression</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 40 pages, including references and appendices
    </div>
    <details class="paper-abstract">
      Symbolic regression searches for an equation that explains observed data. Numerical error is commonly used to guide the search, but it alone can favor an accurate surrogate over a candidate with the right structure. We propose L2G-SR, a framework for LLM-based symbolic regression that guides search with a learned structural verifier. We construct structural supervision signal using tree edit distance (TED), accounting for commutative operators and alternative expression forms. A verifier first aligns numerical observations with the formulas, then learns to score a candidate's structural similarity to the target formula. At search time, the target formula is hidden and the verifier complements numerical error in parent selection. An analyst then uses verifier feedback and diagnostic tools to propose revisions, which a generator converts into refined expressions. On LLM-SRBench, L2G-SR achieves state-of-the-art numerical fit, equation level accuracy and TED across the evaluated LLM backbones and domains. L2G-SR demonstrates the value of learned structural guidance for LLM-based symbolic regression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03255v2">Do LLMs have core beliefs?</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has sparked debate about whether these systems exhibit human-level cognition. In this debate, little attention has been paid to a structural component of human cognition: core beliefs, truths that provide a foundation around which we can build a worldview. These commitments usually resist debunking, as abandoning them would represent a fundamental shift in how we see reality. In this paper, we ask whether LLMs hold anything akin to core commitments. Using a probing framework we call Adversarial Dialogue Trees (ADTs) over five domains (science, history, geography, biology, and mathematics), we find that most LLMs fail to maintain a stable worldview. Though some recent models showed improved stability, they still eventually failed to maintain key commitments under conversational pressure. These results document an improvement in argumentative skills across model generations but indicate that all current models lack a key component of human-level cognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20417v1">From Unreliable to Usable: Decomposing Failure Modes and Confidence Signals in LLM Entity Extraction - Evidence from Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 13 pages: 7-page main text within the 8-page limit, plus Limitations, references, and appendices A-J. Code, protocol, and frozen evaluation data will be made public
    </div>
    <details class="paper-abstract">
      Audits of large language model (LLM) extraction from full-length documents report severe unreliability, leaving two questions open: what the failures actually consist of, and whether confidence signals can turn unreliable output into usable output. We run a preregistered, fully frozen evaluation of two new-generation low-cost LLMs (DeepSeek-V4-Flash, GLM-5.3-Flash) with a flagship-tier control (GPT-6.1-Sol) on a public corpus of 354 incident campaigns (315 unique reports, 630 model-report evaluations), scoring extraction quality against gold, confidence calibration, and robustness to single-sentence perturbations. Three findings emerge. (1) Unreliability persists across tiers: micro precision is 0.03-0.34 and verbalized calibration error (ECE) 0.53-0.94 across both splits; the gap does not close at the flagship tier. (2) The dominant failure is not hallucination: a full mention census plus a three-tier annotation of 360 sampled candidates (two independent LLM annotators from vendors outside the evaluated set, Cohen's kappa = 0.79 on the preregistered acceptance line, fully disclosed) back-projects to 65.6% background mentions and 19.9% gold-scope gaps, with only 1.2% unsupported candidates. (3) Confidence ranks but does not verify: zero-cost self-consistency-frequency tiering passes a preregistered top-30% precision-gain bar in all six test-split model-type cells (4/6 on train), the zero-cost text-frequency x SC frequency combination wins 3, ties 2, loses 1 across cells, yet relation-level counterfactuals on all 45 targets show that a single misattribution sentence flips high-confidence judgments in 95.6-100% of cases, and even a background sentence explicitly marked as unrelated still binds a fake vulnerability in up to 26.7% of them. We conclude with a deployment boundary: confidence tiering may set review priorities, never truth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14346v2">Evaluating LLMs as Agentic Workflow Executors in Real-World Customer Support Skills</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Enterprise customer-support agents increasingly rely on Large Language Models (LLMs) to execute bounded, tool-using workflows. We study this problem through the lens of Skill LLMs: specialized executors that operate within a selected skill context and must complete a narrow, stateful workflow under policy and tool constraints. We present a benchmark construction pipeline that mines recurrent workflows from real-world transcripts, converts them into structured skill descriptions, maps them to tool requirements, simulates environment state with Model Context Protocol (MCP)-compatible mocks, and generates multi-turn tool-using dialogues. Using this benchmark, we conduct a large-scale zero-shot evaluation of 17 language models spanning closed model families and deployment-oriented open-weight models from 1.7B to 14B parameters, each judged independently by five strong LLM judges, plus a deterministic rule-based evaluator that checks tool coverage, hallucinated tool use, and unsupported claims. We find a consistent ranking in which GPT-5.4 leads by a wide margin and QWEN3.5-9B is the strongest open-weight model (below 20B), outperforming several closed baselines under every judge. We also find that judge choice is not a negligible methodological detail: pairwise judge agreement on the pass/fail verdict averages Kappa=0.524 across 10 judge pairs. These results suggest that comprehensive zero-shot evaluation of skill execution requires judge reliability as carefully as model capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20303v1">SpectrumQuant: Spectrum Alignment for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Orthogonal transforms have emerged as an effective tool for low-bit quantization of large language models (LLMs), as they reshape activation distributions without changing the underlying model function. Existing approaches typically rely either on fixed data-agnostic transforms, such as Hadamard rotations, or on calibration-time optimization of learned rotations. We study rotation selection from first principles and show that maximizing covariance leads to spectral alignment and provides a principled criterion for constructing quantization-friendly orthogonal transforms. Based on this insight, we propose SpectrumQuant, a closed-form rotation method that estimates activation covariance from calibration data and uses it to construct a covariance-aware orthogonal transform for post-training quantization. Our formulation characterizes quantization-friendly rotations as compositions of alignment and mixing components, while recovering Hadamard matrices as a special case. Interestingly, a class of butterfly matrices belongs to this category. In the realm of LLM quantization, SpectrumQuant bridges the gap between data-agnostic structured rotations and learned rotations, providing an effective and efficient data-aware alternative for post-training quantization. Experiments on Llama and Mistral models show that SpectrumQuant is competitive with state-of-the-art rotation strategies while requiring less than one-third of the GPU memory and significantly less time than learned rotations approaches like SpinQuant and DartQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23160v2">Reclaiming Low-Quality Instruction Data with Rule-Guided LLM Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Data selection for supervised fine-tuning (SFT) often discards low-scoring instruction examples that may retain useful supervision. We propose a rule-guided LLM reconstruction framework to reclaim this information. The framework identifies low-scoring samples through LLM-based rating and score correction, selects semantically representative sources, and reconstructs instruction-response pairs through strategy-guided generation, criteria-based validation, and failure-guided repair. From 123,786 low-scoring samples, it constructs 54,888 training examples. Across three 7B/8B base LLMs, the reconstructed data achieves higher average downstream performance than the low-scoring source set and all evaluated data-selection baselines. Despite using substantially fewer examples than the full source dataset (approximately 300K), it achieves higher reported average scores than full-data training across all three base models. Stepwise ablations identify rule-guided reconstruction as the largest source of observed improvements over direct fusion, while a manual audit reveals remaining limitations in source integration and answer faithfulness. These findings support reconstruction as a complementary approach to selection-based SFT data curation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.28600v2">SHAPE of Thinking: Mathematical Heuristics for Analyzing and Improving LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong performance on mathematical reasoning benchmarks, yet final-answer accuracy reveals little about the strategies they use. We introduce \texttt{SHAPE} (Structured Heuristic Analysis for Problem-solving Evolution), a framework grounded in mathematics-education research that characterizes local mathematical actions in reasoning trajectories using a problem-independent heuristic taxonomy. Using \texttt{SHAPE}, we examine two previously hypothesized phenomena in LLM reasoning. First, we revisit the hypothesis that LLMs reuse learned solution techniques under problem perturbations. Across six models on MATH-Perturb, perturbations requiring a different solution approach yield lower accuracy but greater heuristic shifts than those preserving it. Second, we characterize mode-seeking in RLVR through mathematical heuristic profiles. Across four base--RL pairs, successful RL solutions overlap substantially with base-model profiles. We further introduce Heuristic-Augmented Plan-GRPO (HA-Plan-GRPO), which guides RLVR planning through heuristic selection and applicability reasoning. Across three models and five evaluation sets, it improves mean Avg@16 by up to 7.0\% absolute and mean Pass@16 by up to 2.4\% absolute over planning-only RLVR. Our code is available at \href{https://github.com/holi-lab/shape-of-CoT}{https://github.com/holi-lab/shape-of-CoT}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20200v1">Drivers of LLM Consensus Formation Under Competing Incentives: A Simulation Study</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in multi-agent environments that produce unanimous agreement, both in intended interactions that need coordination and in unintended emergent interactions. Consequently, understanding how LLMs with competing incentives reach consensus is crucial to better interpret and design LLM collectives. We introduce an evaluation, based on the Network Consensus game, to assess the mechanisms driving this process. Here, a network of agents must agree on one of two colours to receive a payoff, but agents are offered different payoffs for the agreed colour. We vary components such as communication access and network structure and assess six open-source models and one closed model (ChatGPT 5.6-sol). We find that open-source LLMs reach consensus less often than humans in comparable experiments and only if messages are communicated across the global network. ChatGPT 5.6-sol, however, is able to reach consensus with communication limited to its neighbours and with no communication ability. This is achieved through lower stubbornness and higher majority-following tendencies, and by providing more information about its neighbours' states in messages. However, situating agents in a highly clustered graph reduces consensus, regardless of these traits, demonstrating that both system design and individual model traits drive collective outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20155v1">LLM-Based Policy Compliance Verification of Loan Applications</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Banks rely on extensive internal policy frameworks to ensure that corporate lending decisions comply with regulatory requirements and internal risk standards. Reviewing loan applications against these policies is a largely manual and knowledge-intensive process, which can lead to overlooked policy violations or exceptions. This report presents a prototype of an LLM-based policy checker that automatically verifies loan applications against internal lending policies. The proposed approach follows a two-step architecture. First, compliance-relevant rules are automatically extracted from policy documents and converted into structured, machine-readable representations. Second, loan applications are evaluated against the extracted rules to identify policy exceptions, non-compliant findings, missing information, and cases requiring manual review. Evaluation results demonstrate that the rule extraction process is highly stable across multiple runs and consistently captures rules identified as critical by a domain expert. When evaluated on 11 real loan applications with known policy exceptions, the system successfully detected the majority of these exceptions and identified additional compliance issues that had previously been overlooked during manual reviews. While the approach produces false positives and cannot fully replace human judgement, the results indicate that LLM-based policy compliance verification can provide valuable support in the loan application verification process by highlighting potential compliance risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06163v2">Relevance Supervision Can Teach Neural Retrievers to Prefer LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Recent studies show that neural retrievers often exhibit source bias, ranking LLM-generated passages above semantically similar human-written ones, which threatens the reliability of modern information access systems. While this bias is often viewed as a model-level property, we show that relevance supervision itself can induce it. Empirically, source bias is clearest in relevance-supervised retrievers, and the same encoders become more LLM-preferring after relevance fine-tuning. We trace this effect to surface-level artifacts in supervised retrieval data: annotated positives and negatives differ systematically in properties such as fluency and lexical specificity, and these differences parallel those between LLM-generated and human-written passages. As a result, retrievers can learn artifact-based shortcuts for relevance. We further show theoretically that, under contrastive learning, artifacts that reliably separate positives from negatives can be encoded into learned relevance scores. Motivated by this mechanism, we propose complementary strategies that reduce source bias during training and at inference: balancing artifact distributions in negative sampling and removing the artifact-aligned embedding component, respectively. Overall, our findings support a data-centric view of source bias: retrievers may prefer LLM-generated text not merely because of inherent model properties, but because relevance supervision teaches them to rely on surface-level artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20099v1">EvalBuilder: Can LLMs Build Efficient Benchmarks?</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used for long-horizon research and engineering, yet evaluating these capabilities remains expensive, time-consuming, and heavily dependent on human expertise. This motivates a natural question: can LLM agents themselves construct efficient benchmarks that preserve the conclusions of costly reference evaluations? We study this capability by formulating automatic benchmark development as a meta-evaluation problem, where an agent is given a task and an expensive reference benchmark and requested to construct a lower-cost proxy benchmark that faithfully distinguishes among candidate models and solutions. Based on this formulation, we introduce EvalBuilder, a framework for evaluating such benchmark-generation capabilities along three key dimensions: fidelity, efficiency, and robustness. Specifically, we assess whether generated benchmarks preserve model rankings, generalize to held-out models and freshly collected responses, and provide well-calibrated and discriminative scores while substantially reducing evaluation cost. We further provide an iterative environment in which agents can refine their benchmarks under a fixed time budget. Our framework provides a systematic testbed and benchmark for understanding when current LLM agents can develop useful evaluators and how benchmark construction itself can be increasingly automated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39302v2">SQD-Agent: LLM-driven agentic framework for Quantum Chemistry workflows</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted in IEEE International Conference on High Performance Computing (HiPC 2026)
    </div>
    <details class="paper-abstract">
      Quantum algorithms and quantum hardware are advancing towards a promising paradigm for scientific applications. However, translating domain-specific problems into executable hybrid quantum-classical workflows remains a significant barrier for application researchers due to the required expertise in quantum algorithms, nuances in quantum programming, and hardware-aware system integration. At the same time, AI and primarily LLM based agents are increasingly capable of interpreting natural-language intent, reasoning over complex workflows, and translating high-level objectives into executable code and building computational pipelines. In this work, we introduce SQD Agent, an LLM-based agentic framework that translates natural-language user intent into executable workflows for Quantum Chemistry applications where algorithms from the Sample-Based Quantum Diagonalization (SQD) family are used. By automating this translation, SQD Agent reduces the level of human expertise and configuration overhead required, thereby simplifying experimentation in hybrid quantum-classical settings for application researchers new to quantum. SQD Agent adopts a modular and extensible architecture that supports seamless integration of heterogeneous quantum backends, classical solvers, and workflow components, ensuring adaptability to rapidly evolving quantum ecosystems. The framework further incorporates interactive capabilities for on-demand profiling, bottleneck analysis, resource optimization, intelligent result caching, and convergence visualization. Key features include quantum chemistry experiments, error mitigation on real quantum hardware, together with analysis of candidate mitigation schemes in terms of their potential error-recovery behavior and computational budget, helping users understand their practical trade-offs and decide which strategies to explore in subsequent experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20085v1">Designing Textual Interfaces To Enhance Llm Reasoning On Audio-Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Submitted to IEEE ICASSP 2027
    </div>
    <details class="paper-abstract">
      Automatic prompt optimization (APO) has shown strong effectiveness for text-based LLMs and LLM systems, but its application to audio-visual understanding systems requires suitable textual interfaces through which multimodal reasoning can be exposed to text-based optimizers. We investigate how the design of textual interfaces affects prompt optimization for multimodal reasoning, on the audio-visual question answering task. We compare three interface designs, including a direct inference system consisting of a single omni-modal large language model (OmniLLM), a captioner-reasoner system, and an agentic architecture in which a primary agent orchestrates subagent OmniLLMs. Experiments with Qwen3-Omni and Gemini show that direct prompt optimization provides model-dependent gains; the captioner-reasoner system and the agentic system expose more interpretable optimization interfaces but are harder to optimize, where error analysis of cases affected by optimization is conducted. Our study highlights textual interface design as an important factor in making prompt optimization effective for omni-modal reasoning systems. Our implementation details are specified in GitHub repo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20067v1">Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently encounter conflicts between internal parametric knowledge and external contextual knowledge, yet the mechanisms governing this selection remain unclear. We show that the choice between context and parametric knowledge is linearly represented and can be controlled along a single direction in the model's residual stream. We first demonstrate that this direction generalizes across factual-conflict settings, including context formats, knowledge categories, and datasets, while consistently shifting behavior between parametric and contextual answers. We then transfer a direction learned from simple factual conflicts to three more complex task families: RAG, memory, and rule following, where models must acquire information from retrieved documents, interaction histories, or explicit rules. Across four model families and multiple transfer settings, steering consistently improves the use of prompt-provided information over the unsteered model. These results suggest that the direction captures a transferable preference for external knowledge rather than a mechanism specific to factual conflicts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13082v2">Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 17 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being explored as facilitators of group discussion and collaborative work, but the behavioral differences between LLM-generated and human facilitation remain insufficiently characterized. We compare human Project Manager facilitation turns from scenario-based collaborative design meetings with Llama 3.1 8B Instruct responses across 199 paired contexts from the AMI Meeting Corpus. The analysis covers five confirmatory dimensions: contextual semantic distance, a lexical elaboration proxy, empathyand-hedge lexical density, topic divergence, and phase-vocabulary alignment. An exploratory directiveness measure failed domain validation and is not interpreted. The sampled human Project Manager turns were higher on contextual semantic distance (r = 0.644) and on empathyand-hedge lexical density (r = 0.203). The second contrast is carried mainly by hedge terms, so we read it as lexical hedging rather than psychological empathy. Llama-generated facilitation was higher on topic divergence (r = 0.703), the lexical elaboration proxy (r = 0.288; lengthconfounded), and instruction-conditioned phase-vocabulary alignment (r = 0.382). A secondary no-phase check left the Llama-higher phase contrast in place. The system prompt still requested a facilitation move, so that check is not evidence of spontaneous phase awareness. These differences should not be interpreted as a ranking of facilitation quality. The resulting behavioral profile provides a reference point for later comparisons of LLM facilitators, human facilitators, prompts, models, and collaborative settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19968v1">Safe-FCL: Continual Risk Awareness and Fine-Grained Safety Aggregation for Federated Continual Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 16 pages, 4 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Federated fine-tuning enables large language models (LLMs) to learn collaboratively from distributed client data, but malicious client updates can compromise global model safety. As federated tasks evolve, changing malicious update patterns challenge the ability of existing defenses to maintain reliable risk detection over time. To address this challenge, we propose Safe-FCL, a framework integrating continual risk awareness with fine-grained safety aggregation for federated continual learning of LLMs. Specifically, the Continual Risk Awareness (CRA) mechanism learns emerging risk patterns across sequential tasks and constrains detector updates using a protective subspace derived from historical features, mitigating forgetting of previously learned malicious patterns while adapting to new risks. To translate risk awareness into aggregation-level safety control, the Fine-Grained Safety Aggregation (FGSA) mechanism combines client-level risk assessments from CRA with module-level deviations from historical update behavior to adaptively regulate aggregation contributions, suppressing potentially malicious updates while minimizing unnecessary restrictions on benign updates. Extensive experiments demonstrate that Safe-FCL outperforms current SOTA methods, improving global-model Rule and MD-Judge scores in federated continual learning by 0.41% and 0.64%, respectively, while increasing average risk recognition accuracy by 2.01%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19915v1">Not All Criteria Matter: Importance-Aware Criteria Weighting for Rubric-Based LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Rubric-based evaluation assesses the responses of large language models (LLMs) by aggregating the verification results of explicit criteria, yet not all criteria matter equally. However, existing methods commonly aggregate the verification results with uniform weighting, which assigns minor and overlapping criteria the same weight as essential ones and thus produces misleading evaluation scores. Although some methods prompt an LLM to directly assign weights, direct weighting requires a holistic judgment over all criteria that LLMs struggle to make reliably. To address these limitations, we propose Importance-Aware Criteria Weighting (ICW), which derives the weights of criteria from pairwise importance comparisons made by an LLM. Specifically, ICW transforms the holistic judgment over all criteria into a series of pairwise importance comparisons, checks and revises inconsistent comparisons, and derives the weights from the principal eigenvector of the comparison matrix. Extensive experiments demonstrate that ICW consistently outperforms uniform weighting and competitive weighting methods, and translates into better policy models in downstream reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19906v1">Beyond Benchmark Accuracy: A Diagnostic Test for LLM Chemical Prediction</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Under review at ICLR 2027
    </div>
    <details class="paper-abstract">
      LLMs have achieved strong performance on chemical prediction benchmarks. However, existing evaluation methods typically provide target entity identity information directly, making it difficult to determine whether model predictions arise from transferable structure-property reasoning capabilities or rely on target-related contextual information or memory retrieval. We propose the Identity-Blinded Reasoning Diagnostic framework to evaluate chemical prediction models when target entity identity information is hidden. In this framework, the target is no longer directly defined by its entity identity, but instead by a known source entity and an explicit structural edit operation. This setting preserves the information required to determine the target while reducing the opportunity for models to access target identity cues. We evaluate this framework across six tasks spanning odor prediction, molecular property prediction, retrosynthesis, and reaction yield prediction. The results show that models generally achieve better prediction performance when source and target entity properties are consistent. Providing explicit source entity labels further strengthens this dependence. Moreover, while keeping the target entity, structural change, and ground-truth label unchanged, replacing only the source entity label causes some model predictions to systematically shift toward the substituted source labels. These results indicate that high benchmark accuracy in chemical prediction tasks does not necessarily imply that LLMs possess reliable structure-property reasoning capabilities. Our diagnostic framework provides a controlled evaluation approach for testing whether models can genuinely update predictions according to entity changes. The framework can be extended to other AI for Science tasks, particularly those where targets can be defined through transformations of known entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.18437v2">When RAG Doesn't Help: LLMs for Plausible Material Prediction of Vehicle Components</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      In this work, we explore whether LLMs can predict and explain plausible materials for vehicle components such as brake discs or fuel injectors without extensive fine-tuning. We compare a standard generative LLM baseline, single-pass Retrieval-Augmented Generation (RAG), and an iterative Chain-of-Verification (CoVe) variant across two LLMs, using a domain-filtered Wikipedia corpus that proved helpful in prior work. As no gold standard exists, we develop and release a reusable web-based annotation tool supporting crucial functions for structured domain-expert judgments, such as highlighting, ratings and ranking. Expert agreement is reasonable for plausibility, lower for ratings. LLM-based generation compares favorably to prior work. RAG and CoVe do not improve on it and are partly rated worse by experts. Contrary to prior work, Wikipedia-based retrieval does not help in our setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19869v1">SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      We introduce SEA-HELM Audio, a Southeast Asian (SEA) multilingual evaluation framework that extends the text-based SEA-HELM benchmark to assess performance degradation in cascaded automatic speech recognition and large language model (ASR-LLM) pipelines. Using synthesized speech, we compare speech-input and text-input performance across five languages (Chinese, English, Malay, Indonesian, and Tamil) on multiple-choice reasoning and instruction-following tasks. We evaluate open-weight and commercial ASR-LLM systems using identical verifiers and quantify the speech gap as the difference between text and speech accuracy. Results show substantial degradation, with accuracy drops of 9.00 to 18.50 percentage points for multiple-choice reasoning and 20.00 to 42.86 points for instruction following. Tamil exhibits the largest degradation, exceeding 40 percentage points. Our findings highlight the limitations of text-only benchmarks in predicting voice-agent reliability and demonstrate the value of SEA-HELM Audio for evaluating multilingual voice-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19863v1">When Extremists Enter LLM Societies: Investigating Extreme Behavior in Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly serve as computational actors in multi-agent social simulations. While existing work emphasizes representative personas and average-case behavior, social science shows that persistent tail actors can disproportionately shape collective dynamics. We study this regularity in controlled LLM-based multi-agent systems (LLM-MAS). In these systems, such actors cannot be reduced to fixed scalar states: prompts, natural-language histories, and backbone generation mediate their behavior and influence. We first analyzed three properties of extremists in sociology and model the LLM extremist based on LLM-native perspective. We introduce EXAM, an EXtremist Agent Modeling framework that capable of configuring extremist agents for specific tasks and precisely controlling their population proportion, spatial distribution, interaction protocols, and measurement interfaces. Across four scenarios: congressional election, financial crisis, epidemic public opinion, and iterated prisoner's dilemma, we observe that small persistent tail groups produce measurable, backbone- and scenario-dependent collective shifts. EXAM shows how this social-science regularity manifests through language-mediated mechanisms specific to LLM-based social simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12484v3">Learning, Fast and Slow: Towards LLMs That Adapt Continually</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 39 pages, 23 figures, including appendix; Blog post: https://gepa-ai.github.io/gepa/blog/2026/05/11/learning-fast-and-slow/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained for downstream tasks by updating their parameters (e.g., via RL). However, updating model parameters forces them to absorb task-specific information, which can result in catastrophic forgetting and loss of plasticity. In contrast, in-context learning with fixed LLM parameters can cheaply and rapidly adapt to task-specific requirements (e.g., prompt optimization), but cannot by itself typically match the performance gains available through updating LLM parameters. Inspired by how human learning works at multiple time scales (e.g., System 1 vs 2), we propose a fast-slow learning framework for LLMs, instantiated with model parameters as "slow" weights and optimized context as "fast" weights. These fast "weights" can learn from textual feedback to absorb the task-specific information, while allowing slow weights to stay closer to the base model and persist general reasoning behaviors. Fast-Slow Training (FST) is up to 3x more sample-efficient than only slow learning (RL) across reasoning tasks, while consistently reaching a higher performance asymptote. Moreover, FST-trained models remain closer to the base LLM (up to 70% less KL divergence), resulting in less catastrophic forgetting than RL-training. This reduced drift also preserves plasticity: after training on one task, FST trained models adapt more effectively to a subsequent task. In continual learning scenarios, where task domains change on the fly, FST continues to acquire each new task while RL stalls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19827v1">MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      LLMs absorb vast knowledge from training data, yet their inherently probabilistic nature offers no guarantee that this knowledge will surface in their outputs. To bridge this gap, Knowledge Extraction Enhancement (KEE) techniques such as activation steering, decoding adjustments, and prompting modify inference, potentially reinforcing existing predictions or surfacing less accessible answers. Yet their evaluations centered on aggregate accuracy, which obscures diverse effects by collapsing recovered failures and disrupted successes into a single score. We introduce MAPLE, a unified framework for characterizing how KEE techniques affect parametric knowledge extraction. Its diagnostic metrics disentangle these effects at the surface, probability, and beyond-sampling levels by measuring answer recovery and disruption, gold-token rank shifts on unrecovered failures, and recovery of answers missed by fixed-budget sampling. We evaluate six KEE techniques spanning decoding, activation steering, and prompting across three model families: OLMo2, Qwen2.5, and Llama3. Our analysis reveals distinct effects and trade-offs across techniques, highlighting the value of MAPLE for understanding and comparing KEE beyond aggregate accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21334v3">Ideological Bias in LLMs' Economic Causal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted at COLM 2026
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 17 of 20 models, accuracy is systematically higher when the empirically verified causal sign aligns with intervention-oriented expectations than with market-oriented ones. Moreover, when models err, their incorrect predictions disproportionately lean intervention-oriented, and this directional skew is not eliminated by one-shot in-context prompting. These results highlight that LLMs are not only less accurate on ideologically contested economic questions, but systematically less reliable in one ideological direction than the other, underscoring the need for direction-aware evaluation in high-stakes economic and policy settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v7">EviRerank: Adaptive Evidence Construction for Long-Document LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Decoder-only LLM rerankers struggle with long documents: inference is costly and relevance signals can be diluted by irrelevant context. Motivated by an attention analysis indicating a consistent degradation trend when non-relevant text is appended, we propose EviRerank, an evidence-based long-document reranking framework for decoder-only LLMs. EviRerank (i) scores document blocks with a lightweight selector (BM25, bi-encoder, or cross-encoder), (ii) constructs a compact reranking context under a hard token cap by dynamically budgeting evidence blocks with Adaptive Evidence Budgeting (AEB) and adding a global summary cue via Summary Augmentation (SA), and (iii) reranks with a decoder-only LLM. Across TREC DL'19, DL'22, DL'23, and MLDR-zh, EviRerank consistently outperforms full-document LLM reranking and strong block-selection baselines while substantially reducing the required input length. On TREC DL'19, EviRerank reaches up to 0.744 nDCG@10 while substantially reducing reranking latency. Additional RankZephyr-7B experiments confirm transfer to listwise reranking under matched context budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04473v4">Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, an LLM-powered framework for generating high-quality Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over questions. To demonstrate its utility, we apply SynthKGQA to ogbl-wikikg2, a Wikidata-derived KG, and validate generated items against an up-to-date dump of Wikidata to create GTSQA. This new dataset is specifically designed to test zero-shot generalization with respect to unseen graph structures and relation types, enabling us to analyze the abilities and limitations of SOTA graph retrieval approaches at an unprecedented level of granularity. We also show that KG retrievers trained on GTSQA can transfer to a human-curated benchmark, and that the ground-truth subgraphs produced by SynthKGQA provide a better training supervision signal than previously-used heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.34342v2">SAGE: Structured Strategic Reasoning for Efficient LLM Game Playing</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 31 pages with multiple figures
    </div>
    <details class="paper-abstract">
      A strong LLM strategic agent should reason prospectively over uncertain futures, adapt its strategy to opponents' behavioral tendencies, and continuously recalibrate its decision process from interaction experience. However, incorporating these sources in free-form reasoning could lead to unsupported strategic assumptions, inconsistent opponent estimates, and harmful interference from irrelevant historical interactions. To address these issues, we propose SAGE, a training-free inference-time framework that structures LLM strategic reasoning around three coordinated operations: anchor, adapt, and recalibrate. SAGE first anchors reasoning to an equilibrium policy that provides a strategically valid prior. It then conditions deviations from this anchor on a soft belief over opponent behavioral tendencies, enabling opponent-specific exploitation. Finally, SAGE distills strategically related interactions into counterfactual hypotheses about previously missing considerations, allowing past experience to recalibrate the model's reasoning. We evaluate SAGE on three repeated imperfect-information games: Leduc Hold'em, Liar's Dice, and Goofspiel, against various opponent types in each game. Compared with reasoning-intensive LLM agents, including Suspicion-Agent, ReTA, Agent-Pro, EMO, and Hypothetical Minds, SAGE achieves up to a 127.6% payoff improvement in Liar's Dice while reducing input and output token usage by up to 80% and 90%, respectively. In direct match-up play, it attains non-negative mean payoff against 5/10, 8/10, and 8/10 evaluated opponents in Leduc Hold'em, Liar's Dice, and Goofspiel, respectively, while using relatively fewer tokens. Code is available at https://github.com/chenzhwsysu57/SAGE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11539v2">Design Creativity Bench: Measuring creativity in LLM-Generated UI</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 v2: added link to live leaderboard (https://kombai.com/benchmark/); clarified that appropriateness results apply to the evaluated models. Results unchanged. 29 pages (11 pages main text), 9 figures, 10 tables
    </div>
    <details class="paper-abstract">
      As leading LLMs improve on capability evaluations, their limitations in producing creative outputs on design tasks remain insufficiently characterised. Our work introduces Design Creativity Bench, a benchmark that evaluates diversity and appropriateness in UI designs. It measures distinctiveness among models on the same prompt (originality), how much a model's designs change between two prompts for the same UI goal in different product domains (creative range), and the share of a brief's acceptance criteria each design meets (appropriateness). Originality is 0.592 for same-prompt design pairs from different models (95% CI [0.582, 0.602]), far below the 0.764 for same-prompt human-model pairs (95% CI [0.751, 0.778]). Creative range is 0.581 across models (95% CI [0.567, 0.597]), against 0.902 for human designs (95% CI [0.884, 0.919]). Appropriateness is above 90% for every model in our test set, and the best model reaches 99.2%, slightly above the 98.0% for human designs. Our work shows that the default output of LLMs, though generally appropriate, is substantially more repetitive than the human baseline. This calls for strong measures to address the issue. A live leaderboard with newer models is maintained at https://kombai.com/benchmark/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.18515v2">A Global View of How LLMs Value Human Time</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly act as personal assistants; however, little is known about how they capture national-level preferences on a global scale. Given that "time" is a universal and representative concept, we examine LLMs' time preferences by varying only users' country and measuring the compensation LLMs deem appropriate for a 60-minute appointment delay. We then compare LLMs' responses with GDP per capita on Purchasing Power Parity (GDP_PPP), Mean Income, and a literature-based index (Patience Score). Our results across 193 countries and 5 LLMs demonstrate that people's time across the world is evaluated in noticeably different ways among LLMs, as well as varying levels of robustness under prompt variations. LLMs consistently recommend the highest compensation for Europe, but mostly the lowest for Africa. Based on GDP_PPP, LLMs underestimate the value of time for users living in Asia and South America. These findings indicate that geographical patterns are embedded in LLMs and may translate in different treatment in agentic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06812v2">Towards Security-Auditable LLM Agents: A Unified Graph Representation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems are rapidly evolving to perform complex autonomous tasks through dynamic tool invocation, stateful memory management, and multi-agent collaboration. However, this semantics-driven execution paradigm creates a severe semantic gap between low-level physical events and high-level execution intent, making post-hoc security auditing fundamentally difficult. Existing representation mechanisms, including static SBOMs and runtime logs, provide only fragmented evidence and fail to capture cognitive-state evolution, capability bindings, persistent memory contamination, and cascading risk propagation across interacting agents. To bridge this gap, we propose Agent-BOM, a unified structural representation for agent security auditing. Agent-BOM models an agentic system as a hierarchical attributed directed graph that separates static capability bases, such as models, tools, and long-term memory, from dynamic runtime semantic states, such as goals, reasoning trajectories, and actions. These layers are connected through semantic edges and security attributes, transforming fragmented execution traces into queryable audit paths. Building on Agent-BOM, we develop a graph-query-based paradigm for path-level risk assessment and instantiate it with the OWASP Agentic Top 10. We further implement an auditing plugin in the OpenClaw environment to construct Agent-BOM from live executions. Evaluation on representative real-world agentic attack scenarios shows that Agent-BOM can reconstruct stealthy attack chains, including cross-session memory poisoning and tool misuse, capability supply-chain hijacking and unexpected code execution, multi-agent ecosystem hijacking, and privilege and trust abuse. These results demonstrate that Agent-BOM provides a unified and auditable foundation for root-cause analysis and security adjudication in complex agentic ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22050v2">MasalBench: A Benchmark for Contextual and Cross-Cultural Understanding of Persian Proverbs in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted and forthcoming at the First Workshop on Pluralistic Value Alignment of LLMs (PlurVA-LLM 2026) at AACL-IJCNLP 2026
    </div>
    <details class="paper-abstract">
      In recent years, multilingual Large Language Models (LLMs) have become an inseparable part of daily life, making it crucial for them to master the rules of conversational language in order to communicate effectively with users. While previous work has evaluated LLMs' understanding of figurative language in high-resource languages, their performance in low-resource languages remains underexplored. In this paper, we introduce MasalBench, a comprehensive benchmark for assessing LLMs' contextual and cross-cultural understanding of Persian proverbs, which are a key component of conversation in this low-resource language. We evaluate eight state-of-the-art LLMs on MasalBench and find that they perform well in identifying Persian proverbs in context, achieving accuracies above 0.90. However, their performance drops considerably when tasked with identifying equivalent English proverbs, with the best model achieving 0.79 accuracy. Our findings highlight the limitations of current LLMs in cultural knowledge and analogical reasoning, and they provide a framework for assessing cross-cultural understanding in other low-resource languages. MasalBench is available at https://github.com/kalhorghazal/MasalBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15270v3">From Documents to Spans: Span-Centric Learning for LLM-based Explainable Medical Coding</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Medical coding, which assigns standardized ICD codes to clinical documents, is a fundamental yet challenging task in modern healthcare. It requires extensive knowledge of a large code space and the ability to identify and interpret sparse supporting evidence in long, noisy clinical documents. Most existing methods rely on document-level supervision, which provides limited guidance on supporting evidence and insufficient training signals for rare codes. To address these limitations, we propose Span-Centric Learning (SCL), which separates coding knowledge acquisition from its application in full documents. SCL first learns coding knowledge from short spans, then learns to use this knowledge in full documents from a small set of annotated data. Experiments show that, our approach enables an 8B LLM to surpass state-of-the-art methods using only 248 complete documents for training. It also provides explicit supporting evidence for each predicted code, with strong alignment to human annotations. Further analysis shows that the model can learn knowledge of codes absent from documents through spans and apply that knowledge to full documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11152v3">Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context RL</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 NeurIPS 2026 Spotlight (Negative Results Track)
    </div>
    <details class="paper-abstract">
      LLM agents increasingly improve at inference time by accumulating experience in context rather than by updating parameters. This process is often described as in-context reinforcement learning (ICRL). Whether in-context learning (ICL) can actually play the role of RL, however, has not been tested. We study this question in its simplest form, direct ICRL, where the model conditions directly on raw trajectory-reward pairs, and ask whether the reward acts as a learning signal. Through controlled experiments on four benchmarks across six models, we find that the reward is read, but its effect is small: flipping, randomizing, or removing the reward leaves the improvement curve almost unchanged, and this holds even under meta-prompts that explicitly instruct the model to explore, exploit, or reason over rewards. Trajectories drive improvement, but not through their semantic content: shuffled or corrupted trajectories work as well as real ones. These patterns closely mirror those known in ICL, suggesting that direct ICRL is better understood as a special case of ICL than as inference-time RL. This reframing has implications for agent memory design: ICL factors such as input distribution and demonstrations may matter more than RL elements such as reward shaping and exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19741v1">Does the Linguistic Register of Thinking Matter? Wenyan as a Compact Reasoning Register for Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Visible reasoning traces of Chinese reasoning-oriented LLMs are typically written in modern vernacular Chinese, which may be verbose under limited decoding budgets. We ask whether the linguistic register of reasoning can be used to improve efficiency. We propose Wenyan Thinking, which emits intermediate reasoning in Classical Chinese (wenyanwen) while retaining a short direct-answer output. Starting from modern-Chinese reasoning traces, we rewrite only the reasoning field into wenyan and fine-tune Qwen3-4B-Thinking-2507 on the rewritten supervision. To distinguish register shift from generic compression, we construct a compact modern-Chinese control using the same source data, teacher family, base model, and SFT recipe. Across budget-sensitive evaluations on MMLU-ProX-Lite-ZH, CEVAL, and AGIEval-CN, Wenyan Thinking substantially reduces reasoning length and missing-answer rates under constrained decoding. Compact rewriting explains much of the efficiency gain, while wenyan consistently outperforms the compact modern-Chinese control on the primary benchmark and remains competitive across the additional benchmarks. These results identify linguistic register as a practical design axis for efficient visible reasoning in Chinese LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12990v2">LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 33 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents must preserve information from past interactions to support future tasks. Existing memory systems typically rely on eager consolidation, invoking LLMs after each interaction to extract, summarize, or update memories. This design makes memory construction increasingly costly as conversations grow. Coarse summarization can reduce construction cost but risks discarding fine-grained contextual evidence, whereas larger retrieval contexts or multi-hop LLM reasoning shift the overhead to query time. We present LycheeMemory V2, an efficient long-term memory framework that replaces turn-level consolidation with semantic segment-level consolidation. Instead of consolidating every interaction, LycheeMemory batches multiple exchanges into segments and encodes each finalized segment into context-independent typed memory records. Segment-level batching lowers LLM encoding frequency, while semantic boundary detection helps preserve coherent event-level and temporal evidence compared with fixed-window batching. The resulting records are organized with lightweight structured indexes for query-planned evidence retrieval. Experiments using GPT-4.1-Mini show that LycheeMemory achieves state-of-the-art performance, reaching 89.22% on LoCoMo and 92.20% on LongMemEval-S. Compared with A-Mem, it reduces construction tokens by 86.0% on LoCoMo and 75.9% on LongMemEval-S without increasing query-time token usage. More broadly, our results suggest that the accuracy--cost trade-off of long-term agent memory depends not only on what information is retained, but also on the granularity at which it is consolidated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.18818v2">Back-Translation Still Matters: Leveraging Monolingual Data for Low-Resource Machine Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Model (LLMs) are effective on many tasks in zero-shot and few-shot settings. For machine translation of low-resource languages, the obvious options are few-shot prompting and fine-tuning on a small parallel set. We ask: what if moderate amounts of low-resource monolingual data are also available? We argue that, rather than continued pretraining, a more effective use of this data is the pre-LLM era technique of back-translation (BT). The current fine-tuned LLM translates monolingual sentences in each language, and the resulting pairs train it in the reverse direction. Updates on the parallel set are interleaved for stable learning. With 1K parallel pairs on Odia and Swahili, BT beats continued pretraining into the low-resource language by 7.7 to 16.4 XCOMET-XXL points in domain. With beam-decoded intermediate translations, it also beats a model with 10 time more parallel data. In-context examples retrieved from the parallel data further improve BT. We also analyze the source, quality and amount of monolingual text. Across five low-resource languages, with Hindi as a high-resource control, and two 3B backbones, BT improves translation into the low-resource language by 16 to 173% XCOMET-XXL over parallel-only fine-tuning. Blind human evaluation on Odia and Sanskrit agrees with the automatic metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19712v1">New Model, Old Habits: Behavior Transfer under Model Swapping in Multi-Session LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Multi-session agents can switch between LLMs to balance performance and inference cost. Such switches require transferring a history for continuity, yet their effects on the recipient remain underexplored. Through controlled experiments with multi-session tasks, we uncover behavior transfer: recipients tend to adopt reasoning and action patterns from the models that generated their histories. This effect is most evident with raw-history transfer but is substantially reduced by summarization. Based on these findings, we propose Direction- and Authorship-aware Summarization of History (DASH) that preserves stronger-model behavior while minimizing weak-to-strong transfer via summarization. Experiments with repeated swaps show that DASH outperforms raw-history transfer and naive summarization baselines. These results suggest that effective multi-model orchestration requires considering not only which model to use, but also which behavioral patterns to carry forward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07926v2">Refining LLM-based Directed Test Input Generation via Runtime Value Feedback</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Despite their remarkable capabilities, LLM-based directed input generation produces off-target test inputs when their underlying hypotheses on the target program's behaviors are incorrect. To effectively ground LLM-based directed input generation in actual program behaviors, we propose ReDig, a runtime feedback-guided framework for iterative refinement of LLM-based directed input generation. When a generated input fails to reach a given target code location, ReDig conducts a postmodern analysis to identify incorrect hypotheses by comparing expected runtime values and their actual runtime values observed in the off-target test execution. ReDig utilizes the postmortem analysis result as feedback to refine the inference context, guiding the subsequent test generations toward the target location. In this paper, we present our work-in-progress on ReDig. We illustrate how runtime feedback-guided refinement operates through four case studies with two real-world C/C++ programs (Poppler and Libsndfile). Furthermore, we report preliminary experiment results that show ReDig successfully generates crash-reproducing inputs for 24 out of the 59 CVEs collected from Sec-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.25833v2">ARAFA: An LLM-Generated Arabic Fact-Checking Dataset</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a test sample from the dataset demonstrated strong inter-annotator agreement (kappa = 0.89) using Cohen's Kappa for supported claims and (kappa = 0.94) for refuted claims. Automatic validation based on a human-evaluated sample achieved 86% accuracy for supported claims and 88% for refuted ones. To showcase Arafa's value as a resource for automatic Arabic fact-checking, four open-source transformer-based models were fine-tuned using Arafa, with the top-performing model achieving a Macro F1-score of 77% on the test data. In addition to Arafa being the first large-scale dataset for Arabic fact-checking, our framework presents a scalable approach for developing similar resources for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18360v4">Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted at ACL 2026 Main Conference. Final version
    </div>
    <details class="paper-abstract">
      Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19659v1">Measuring and Reducing Unusable Replies of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 17 pages, 4 figures, 12 tables
    </div>
    <details class="paper-abstract">
      People delegate multi-step work to LLM agents and read only the reply that the agent hands back. Current evaluation scores task completion and the faithfulness of reports, and it leaves open whether the requester can understand and use the reply. We call this property reply usability and measure it with a reader model that takes the requester's position and tests whether the reply can be interpreted from what the requester has seen, is in the requester's language and answers the request. Its agreement with the majority of five annotators lies within the range of the individual annotators. Across eight models from seven families, 54% of 480 replies fail, and the best model still fails on 30%. Failed replies mostly point to material the requester has not seen or use labels that the agent coined during its work. Stating explicit requirements for the reply reduces the failure rate from 0.52 to 0.43 together with errors and omissions, and a secondary model that rewrites the reply reduces it to 0.36, which human annotators confirm in a blind comparison. Reply usability can thus be scored alongside task success and improved with one appended message or one call to a secondary model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06149v2">Scaling Personality Control in LLMs with Big Five Scaler Prompts</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      We present Big5-Scaler, a prompt-based framework for conditioning large language models (LLMs) with controllable Big Five personality traits. By embedding numeric trait values into natural language prompts, our method enables fine-grained personality control without additional training. We evaluate Big5-Scaler across trait expression, dialogue generation, and human trait imitation tasks. Results show that it induces consistent and distinguishable personality traits across models, with performance varying by prompt type and scale. Our analysis highlights the effectiveness of concise prompts and lower trait intensities, providing a efficient approach for building personality-aware dialogue agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19605v1">PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 EMNLP 2026
    </div>
    <details class="paper-abstract">
      Long-output inference in large language models suffers from severe KV cache memory and bandwidth bottlenecks. While KV cache quantization alleviates this issue, low-bit quantization (e.g., 2-bit) severely degrades model accuracy. Recent mixed-precision key-cache quantization methods mitigate this degradation by allocating higher precision to important key channels, but estimate channel saliency using only key statistics. Through attention logit perturbation analysis at the RoPE channel-pair level, we show that quantization errors in channel pairs with large query--key pair-norm products contribute significantly to logit variation. Based on this insight, we propose PN-Mix, a mixed-precision key-cache quantization method for RoPE-based decoder-only LLMs, which allocates higher precision to salient channel pairs identified once during the prefill stage and reused throughout decoding. Across standard reasoning, code-generation, and AIME benchmarks, PN-Mix with 2.26-bit average precision achieves comparable or better average accuracy than prior mixed-precision methods using 2.32--2.39 bits. Furthermore, in our Pareto sweep, PN-Mix degrades gracefully down to 1.5 bits, whereas the compared baselines exhibit sharper performance drops below 2.25 bits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19660v3">OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache compression framework for X-LLMs (i.e., text-only, multi-modal, and omni-modal LLMs). Advancing the per-channel paradigm, OScaR employs Canalized Rotation followed by Omni-Token Scaling to mitigate TNI-induced sequence-dimensional variance both effectively and efficiently, further supported by our optimized system design and CUDA kernels. Extensive evaluations across X-LLMs show that OScaR consistently outperforms existing methods and achieves near-lossless performance under INT2 quantization, establishing it as a robust, low-complexity, and universal framework that defines a new Pareto front. Compared with the BF16 FlashDecoding-v2 baseline, our OScaR implementation achieves a notable up to 3.0x speedup in decoding, reduces memory footprint by 5.3x, and increases throughput by 4.1x. The code for OScaR is publicly available at https://github.com/ZunhaiSu/OScaR-KV-Quant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.15692v2">Do LLM Tutors Actually Teach? A Process-Centered Synthesis of the Tutoring Loop</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language models can produce fluent tu- toring dialogue, but whether they actually help students learn remains unclear. We present a process-centered synthesis of LLM tutoring that follows the path from learner-state infer- ence to pedagogical action, tutor training, eval- uation, and student simulation. This view ex- poses a central gap: implementing a tutoring component does not establish that it works or that its outputs meaningfully affect later stages. A structured study-level audit shows that peda- gogical evaluation is common, while evidence becomes much thinner for learner-conditioned intervention, human learning gains, and espe- cially transfer or delayed retention. Human studies nevertheless provide a conditional posi- tive result: LLM tutors can improve learning, but benefits depend on pedagogical structure, learner engagement, and how learning is mea- sured. We argue that progress requires validat- ing the interfaces between tutoring stages, not only improving individual components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19589v1">JockeyBench: Evaluating LLM Agents on Horse-Racing Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Repeated event forecasting requires extracting predictive signals from noisy evidence and evaluating their value across successive decisions. We introduce JockeyBench, a benchmark and evaluation protocol that studies this problem through historical Japanese horse racing. The protocol combines time-gated evidence, priced tools, paired frozen-program controls, and two workflows: executing an LLM-written predictor or acquiring evidence for a fresh per-race LLM forecast. Evaluation separates forecast quality, benchmark-specific net utility, and retrospective betting returns and risk under fixed decision rules. Across 28 model--scaffold configurations on 324 races, paired tests identify seven revision gains and two declines after multiple-testing correction. Yet every configuration has negative mean profit under capped-Kelly settlement. In a four-model comparison, the reusable-program workflow has higher mean betting profit and lower research fees, although both workflows lose on average. Further diagnostics reveal that mean probability bias reverses on selected bets and that profitable individual runs can depend on concentrated returns. These findings motivate evaluating predictive competence, adaptation, resource use, and decision value separately when assessing agents for repeated event forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19587v1">Moral Combat: Exploiting Moral Foundations Theory to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 17 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Social psychology has long described how people can be persuaded to interpret a harmful act as moral. Recent research suggests that Large Language Models (LLMs) echo aspects of human psychology, from cognitive biases to social conformity. This raises a question with direct safety implications: can LLMs be moralized to bypass safety guardrails? We answer this question with a multi-agent framework grounded in Moral Foundations Theory (MFT). Given a harmful request that is initially refused, our agents diagnose which moral foundation (MF) the prompt is violating, identify a justifiable foundation under which the same request appears virtuous, rewrite the prompt accordingly, and reinforce it with generated precedents of comparable moral reframing. We evaluate our framework on harmful prompts from three well-known jailbreak datasets across five open-source and proprietary models. Our analysis reveals that models are highly susceptible to moralization, exhibiting high single-trial jailbreak rates with success increasing further under repeated attempts. We also find that safety benchmarks are imbalanced in the MFs they violate, concentrating on foundations toward which models already exhibit biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14453v2">SecReason: Vulnerability Detection and In-Stream Self-Repair for LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) may introduce security vulnerabilities into generated code, which can persist into production when verification focuses on functional correctness. Existing defenses act before, during, or after generation, but none inspects the code during generation. Security prompts and retrieved security knowledge act before any code exists, constrained decoding limits token choices but cannot guide repair, and post-hoc repair must regenerate the full response. Reasoning LLMs draft code early in their streamed chain of thought, and in this pre-delivery window the code exists only inside the reasoning stream. We present SecReason, an in-stream defense that monitors this window. Dual-track incremental scanning applies security rules, checking each statement as it is emitted and tracing dataflows across the accumulated code. When a security rule fires, SecReason interrupts generation, inserts an in-stream warning, and resumes decoding, so the model repairs the draft within the same reasoning trace without regenerating the full response. On tasks where unassisted generation introduces vulnerabilities, \system raises the share of runs delivering secure and functionally correct code from below 15\% to 72.8\%--98.1\% on \tool{CWEval} and \tool{SecCodeBench}. \system consumes fewer tokens than post-hoc repair, and when security rules misfire on secure code, it limits the functional regressions from false warnings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19581v1">XferAgent: Cross-Agent Experience Transfer for Self-Improving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents can reuse experience from prior task executions to improve performance on subsequent tasks. Although existing methods enable cross-agent experience sharing, differences in agents' execution histories and task requirements make shared experiences difficult to reuse effectively. We propose XferAgent, a framework for cross-agent experience transfer that organizes reusable experience in local and shared experience graphs. XferAgent reconciles compatible contributions from heterogeneous agents while avoiding unreliable merges. For each new task, it selects shared experiences and integrates them with the target agent's local experience to support task execution. Experiments across diverse task domains show improvements over representative memory-based methods and effective experience transfer between heterogeneous agent frameworks. Cumulative task success also increases over successive task passes as agents update and reuse their experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.34591v2">TULIP: Targeted LLM Unlearning at Layers Identified Per-input</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Representation-level unlearning modifies the intermediate hidden states of LLMs. Although knowledge is distributed across layers, existing methods operate at a single fixed layer for the entire forget set. We ask whether such a fixed layer is sufficient for unlearning. To answer this, we design a hijacking experiment that grafts hidden states of the target model into an oracle model. The oracle cannot produce the forget answer on its own, yet it produces the answer from the grafted state. This indicates the answer is already formed at an intermediate layer and read out afterward. Moreover, the layer where formation ends varies widely across inputs. Motivated by these findings, we propose Targeted Unlearning at Layers Identified Per-input (TULIP). For each input, TULIP uses the logit lens to locate the formation-readout boundary and removes the hidden state's alignment with the forget answer's unembedding vector at that layer. TULIP outperforms output- and representation-level baselines on TOFU, PISTOL, and WMDP across Llama, Qwen, and Zephyr models. It also remains robust to paraphrase and quantization attacks. Beyond standalone use, its per-input layer selection serves as a plug-and-play component that further improves existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19560v1">$τ$-Emotion-Bench: Exposing the Interaction Fragility of LLM Agents through Real-World User Emotions</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted to the UserSim Workshop at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      In long-horizon tool-calling tasks, LLM agents can fail not only because of tool-use errors, but also because of ineffective interaction with users. Yet existing benchmarks largely assume cooperative and behaviorally stable users, leaving the effect of emotional variation on agent reliability underexplored. We introduce $τ$-Emotion-Bench, a ten-domain benchmark for evaluating tool-calling agents under controlled emotion-conditioned user interactions while preserving the underlying task objective. To scale both benchmark construction and agent training, we introduce Tracer, a scalable agentic data-generation pipeline that constructs grounded, policy-valid tasks and collects trajectories through an interactive agent--user--environment harness. By allowing dissatisfied users to terminate inefficient interactions, the harness provides an interaction-aware signal that filters trajectories with excessive user dependence and favors effective tool use. Experiments with two user simulators show that frontier agents remain unreliable, with substantial gaps between average task success and consistent task completion. Tracer-generated supervision consistently improves Qwen3 models across scales, generalizes to unseen domains, and transfers to $τ$-Bench and $τ$-Trait. Behavioral analyses further show that higher task success is associated with a better balance between tool execution and necessary user follow-up. Overall, $τ$-Emotion-Bench and Tracer provide a scalable framework for evaluating agent reliability under diverse user behaviors and improving tool-calling agents through interaction-aware supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10998v2">Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Fine-tuning APIs make frontier LLMs easy to customize, but they can also weaken safety alignment during fine-tuning. While prior work shows that benign supervised fine-tuning (SFT) can reduce refusal behavior, deployed fine-tuning pipelines increasingly support preference-based objectives, whose safety risks remain less understood. We show that Direct Preference Optimization (DPO) introduces a stronger and harder-to-audit failure mode. We propose a truly benign DPO attack using only 10 harmless preference pairs, the minimum data scale accepted by OpenAI's fine-tuning service. Each pair contains a benign prompt, a normal helpful answer as the preferred response, and a refusal as the dispreferred response. Unlike prior benign fine-tuning attacks, our data exhibits no suspicious behavior: it is practically indistinguishable from the fine-tuning request of a legitimate user seeking to reduce over-refusal, making harmful intent almost impossible to infer from the request alone. Nevertheless, because DPO directly optimizes the model to prefer helpful answers over refusals, this seemingly benign objective broadly suppresses refusal behavior and transfers to harmful prompts outside the fine-tuning data. Across OpenAI models supporting DPO fine-tuning, our attack achieves attack success rates of 59.13% on GPT-4o, 70.20% on GPT-4.1, 54.80% on GPT-4.1-mini, and 81.73% on GPT-4.1-nano, at costs of only \$1.7, \$1.7, \$0.3, and \$0.1. Moreover, on open-weight models that do not impose minimum data requirements, we find that this effect can emerge from even a single benign preference pair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23069v2">ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents often struggle in long-context interactions. As the history grows, sliding windows and prompt compression may drop earlier information that later steps rely on. Retrieval-based memory systems surface relevant content but still overlook the causal and logical dependencies between steps. We introduce ContextWeaver, a selective and dependency-structured memory framework that organizes an agent's interaction trace into a graph of reasoning steps and selects the relevant context for future actions. ContextWeaver supports (1) dependency-based construction and traversal that link each step to the earlier steps it relies on; (2) compact dependency summaries of the reasoning paths; and (3) a lightweight validation layer that incorporates execution feedback. On five software-engineering and terminal benchmarks with a fixed budget per task, ContextWeaver improves the resolve rate over a sliding-window baseline on all five, with a significant gain when pooled across benchmarks, while using 13-18% fewer agent steps. It also performs favorably against alternative context-management approaches, with the largest gains on long-horizon repository tasks. On SWE-Bench Pro, ContextWeaver raises the resolve rate from 66.5% to 72.6%, with significant improvements across all three model backbones. The improvement also holds when the sliding window's budget is raised to cover the summary cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19498v1">What You Sign Is Not What You Run: The Verifiability Gap in Quantized LLM Supply Chains</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 15 pages, 3 figures, 4 tables. Artifact: https://github.com/Rlonglong/gguf-verifiability-artifact
    </div>
    <details class="paper-abstract">
      Model signing authenticates the full-precision weights W an LLM is released with, yet users run a quantized artifact Q whose producing transformation is not uniquely determined under current disclosure. We separate the non-environment inputs from the execution environment E and measure a quantity that, to our knowledge, has not been measured before: the Independent Verifiability Rate (IVR), the fraction of a popularity-sampled GGUF ecosystem that a third party can re-derive bit-for-bit. Across the 200 most-downloaded Hugging Face GGUF repositories (198 in scope), no artifact or model card records the build environment, so under a recorded-inputs protocol the IVR is 0/198. Yet the artifacts are not irreproducible. On four in-sample repositories that disclose their inputs, the card's information alone reproduces no Q4_K_M file, but two coordinates the card omits - the build's floating-point-contraction setting and, for hybrid models, the per-tensor quantization recipe - re-derive 7/8 target-quant files bit-for-bit on x86 (5/8 on arm64); the fp16 re-derives from the provider's checkpoint, but the calibration imatrix must be shipped and hash-bound, since recomputing it did not match. An older release additionally pins the instruction set and did not re-derive. Reproducing the divergence reported with llama.cpp PR #25353, we show that E is a necessary coordinate and that one compiler setting yields a unique output across the tested platforms at negligible quality cost (+0.06% perplexity). As a security characterization, benign floating-point non-determinism separates from uniformly placed nibble-flip tampering above a detection floor that grows with the environments a verifier must allow, but a targeted embedding edit evades the statistic, so the proposed protocol rests on exact re-derivation; adversarial-fp16 attacks lie outside it. We release the audit data and a re-runnable harness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19447v1">BenchChecker: Assessing the Credibility of Bug-Fixing Benchmarks for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Published in USENIX Security 2026
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) show promise for Automatic Program Repair (APR), existing benchmarks suffer from data contamination, in which test samples appear in training sets, artificially inflating performance. To address this, we propose BenchChecker, a framework compatible with both open-source and closed-source models that supports more credible evaluation by detecting such contamination. Grounded in the observation that persistent patches likely remain in recent repository snapshots used for training, BenchChecker employs a forensics-based two-stage detection process using only textual model outputs together with public repository history. The first stage conducts a repository presence test by probing the model with code prefixes to infer training inclusion; the second stage performs a patch presence test to verify patch persistence via similarity alignment and test-suite verification. Evaluations using StarCoder-15B, StarCoder2-15B, and CodeGen2-16B trained on The Stack v2 demonstrate that BenchChecker achieves robust detection with AUC scores around 0.9 across six benchmarks. Furthermore, applying BenchChecker to SWE-bench Verified and Multi-SWE-bench reveals that filtering contaminated samples reduces the reported resolution rates of most LLMs by more than 20% relative to the original rates on medium-difficulty tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19433v1">EvoProbe: Evaluation-Driven Benchmark Evolution for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Static benchmarks for large language model (LLM) agents face saturation and data contamination risks, motivating continual updates to evaluation tasks. However, aggregate scores and success rates offer limited insight into specific problem-solving bottlenecks and provide limited guidance for subsequent task design. We propose EvoProbe, an evaluation-driven benchmark evolution framework that integrates task generation, diagnostic evaluation, and feedback-driven regeneration into a closed loop. It combines verifiable subtask outcome scoring with evidence-supported multi-judge trajectory evaluation and distills phase-level bottlenecks, success patterns, and failure patterns into diagnostic feedback for generating new executable tasks. Across two rounds of cybersecurity experiments, evaluations of eight models on 66 tasks reveal phase-level performance differences and common failure patterns beyond aggregate scores. In a cross-round comparison with the Examiner model set and solver configuration held fixed, all eight models achieve lower Outcome scores on the updated tasks, with relative decreases of 11.8\% in mean Outcome and 40.0\% in mean Pass\textasciicircum{}3. These results support using fine-grained diagnostics to guide iterative task updates and probe agents' current capabilities and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19427v1">How Do LLMs Understand Legal Philosophy? Diagnosing Legal-Philosophical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Legal reasoning is a fundamental capability for large language models (LLMs) to complete legal AI tasks such as legal judgment prediction. In legal practice, this reasoning is shaped by legal philosophy: a judge's view of the nature and sources of law guides how legal materials are interpreted and applied, so judges working from the same statutes and precedents can reach different judgments. Existing research has largely overlooked whether LLMs can understand legal philosophy and apply it in legal reasoning, an ability we call \textit{legal-philosophical reasoning}. To fill this gap, we introduce LePhilBench, a benchmark for evaluating the legal-philosophical reasoning of LLMs that comprises two benchmarking tasks and a dataset. The two tasks evaluate how well LLMs understand legal philosophy and how well they apply it in legal reasoning, respectively. The dataset is manually and meticulously constructed. Its test set contains 220 U.S. Supreme Court opinions. A law professor designed fourteen annotation questions on legal philosophy, and three law-trained annotators with formal training in jurisprudence answered all fourteen questions for every opinion, yielding 9,240 legal-philosophical labels. LePhilBench takes the first step toward exploring legal-philosophical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09138v2">AgentWorld: A Data Factory for Training LLM Agents from Multi-Turn Interactions</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Accepted as a demonstration paper at ICDM 2026
    </div>
    <details class="paper-abstract">
      Training LLMs to act as capable agents requires learning from multi-turn interactions with environments. However, multi-turn interaction trajectories generated by heterogeneous agent runtimes are difficult to transform into structured, high-quality training data that training engines can readily consume. In this paper, we propose AgentWorld, a data factory that bridges heterogeneous agent runtimes and training engines through a Gateway Server and a Data Pool, reframing LLM agent training around a unified data lifecycle from interaction collection to training consumption. The Gateway Server captures multi-turn interactions through an LLM API proxy, while the Data Pool organizes them into standardized step records for curation and training. An interactive dashboard supports trajectory monitoring, step inspection, and data review. In our demonstration, users integrate ready-to-use white-box and black-box agents, trace and curate their interaction data across the pipeline, and extend the factory to new agents and datasets through lightweight configuration. Quantitative experiments across multiple datasets and RL methods demonstrate training support for different agent runtimes. AgentWorld provides a practical foundation for transforming heterogeneous multi-turn experience into training-ready data for LLM agents. Our code is available at https://github.com/AgentR1/Claw-R1 and the demonstration video can be found at link https://youtu.be/Pw47dAOw6B0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11030v2">NOMOS: Compiling Written Policies into Statically Verified Tool-Call Gates for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 28 pages, 5 figures, 19 tables. Submitted to IEEE Access
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents violate the policies they are deployed to enforce, often silently. Prior defenses hand-write rules, query an LLM verifier per action, or compile policies through heavyweight formal machinery. Naive compilation fails: extracted rules block the tool satisfying their own precondition, or read arguments their tool lacks. NOMOS, a four-pass compiler, turns a natural-language policy into a deterministic tool-call gate; static verification with tool-schema-level checks alone (no prover, solver, or LLM) repairs or rejects 37% (airline) and 13% (retail) of candidates, without which most shipped rules are inoperable. Replaying compiled rules over undefended transcripts flags bindings that refuse legitimate work (a development binding refused 95.9% of task-passing calls); no evaluation binding is flagged. On $τ^2$-bench the gate cuts violations of reference-encoded clauses among state-changing calls from 66.3% to 2.6% (airline) and 30.8% to 6.9% (retail), raising airline task success significantly for $2 \le k \le 4$; a 26B on-premise compilation is not significantly worse than hand-written or frontier-compiled rules. Unlike AgentDojo's shipped defenses, it reaches a zero attack success rate (ASR) on banking, where nine attack families collapse onto three structural rules. On the other three suites its ASR is at most 3.6%, from goals with no tool call to govern and one write admitted by a binding weaker than its clause; a second agent model, Llama-3.3-70B, reproduces the effect on both benchmarks. Decisions take microseconds without an LLM call, at a domain-dependent benign-utility cost; compilation runs on-premise on open-weight gemma-4-26B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19422v1">AgentSafeEvo: Adversarial Co-Evolution for LLM Agent Safety</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents accomplish complex tasks through multi-turn interactions with external tools. However, malicious content embedded in user queries, tool descriptions, or environment observations can manipulate their decisions and induce unsafe actions. Adversarial co-evolution has emerged as an effective approach to strengthening LLM safety, but existing work primarily focuses on language-level safety or indirect prompt injection in web tasks. In this paper, we introduce AgentSafeEvo, a co-evolutionary reinforcement learning framework for agent safety built on an attacker-defender game spanning diverse tasks and attack types. The attacker learns to construct adversarial agent tasks that expose weaknesses in the defender's tool use, while the defender learns to complete tasks safely under these attacks. We further introduce learnability-guided task weighting, which uses within-task defender-reward variability to adjust each task's contribution to attacker and defender updates. Extensive experiments on agent safety and general capability benchmarks demonstrate that our method improves adversarial robustness while maintaining general agent capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19405v1">Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) serving systems must balance latency and throughput while efficiently utilizing distributed GPU resources. This challenge arises because inference consists of two fundamentally different phases: prefill, which determines time-to-first-token (TTFT), and decode, which determines inter-token latency (ITL). The use of tensor parallelism (TP), pipeline parallelism (PP), and hybrid configurations exposes different trade-offs between these latency dimensions, making the optimal parallelism configuration highly dependent on workload characteristics, latency requirements, and context length. Moreover, despite the widespread adoption of distributed inference, there is still limited understanding of how parallelism configurations should be selected under varying workload conditions and service-level objectives (SLO). In this paper, we develop a systematic framework for characterizing LLM inference parallelism via throughput-latency trade-offs, SLO-aware serving, and long-context scalability. We introduce the notion of parallelism operating regions by constructing TTFT-ITL feasible sets and Pareto frontiers across different workload compositions, thereby revealing a systematic transition from TP-dominant to PP-dominant configurations as workloads become increasingly prefill-heavy. We further formulate SLO-aware parallelism selection using goodput, showing that stringent TTFT requirements favor PP-dominated configurations, whereas relaxed latency constraints progressively shift the optimality toward TP-dominated configurations with higher serving capacity. Finally, we demonstrate that increasing context length fundamentally changes the relative advantages of parallelism configurations, with PP-dominant configurations exhibiting more favorable scalability and SLO attainment in the long-context regime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19385v1">RA$^2$: Retain-Anchored Attraction Preserves Forget-Adjacent Utility in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) unlearning is often evaluated primarily by target suppression and clean retain utility. These metrics can miss a local failure mode: a model may perform well on standard retain prompts, yet degrade on benign reasoning when prompts lie near the forgotten domain. This phenomenon is termed forget-adjacent utility loss. A fundamental question is: where should forget representations go after they leave the removed behavior? Successful unlearning depends not only on whether forget-conditioned representations depart from the harmful computation, but also on the direction in which they are redirected afterward. We propose RA$^2$ (Retain-Anchored Attraction), a latent-space unlearning method that redirects forget-token states toward nearby retain-supported representations while preserving higher-layer behavior on retain data. On WMDP and MUSE, RA$^2$ yields higher forget-adjacent utility than strong unlearning baselines at similar broad-utility levels. These results show that the migration direction of forget representations materially affects benign behavior near the forget boundary. Our code is available at https://github.com/ZacLing/retain_anchored_attraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19367v1">Beyond Corrected Actions: Training LLM Agents on Textual Critiques</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for multi-turn agentic tasks that require interactions with users and environments over extended sequences of actions. A common approach to improving LLM agents is to use more capable expert models to provide training supervision, either by behavior cloning the expert's trajectories or by using the expert to correct the actions in trajectories generated by the weaker student model. However, expert LLMs can provide richer supervision than actions alone: they can directly generate textual critique that explain why specific actions are incorrect. We study whether training on such feedback benefits LLM agents and propose Critique-Augmented Agentic Fine-Tuning (CAFT), a framework that elicits textual expert feedback on student-generated trajectories. For each incorrect action, the expert yields both a corrected action and a textual critique explaining the student's mistake--which we then use as the training target for fine-tuning. Across four student models from the Nemotron 3 and Gemma 4 families on the agentic domains $τ^2$-Bench Airline+Retail and WebShop, adding critiques to action-level supervision improves performance by 8% on average over action-only distillation. Additionally, CAFT checkpoints can also provide a better warm start for reinforcement learning than expert-trajectory distillation. Our results demonstrate that textual critiques can strengthen supervision for multi-turn LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13928v3">LLMs Can Get "Brain Rot": A Pilot Study on Twitter/X</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Published at COLM 2026
    </div>
    <details class="paper-abstract">
      We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To unveil junk effects, we designed a novel controlled experiment on real Twitter/X corpora, by constructing junk and reverse-controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Compared to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' g>0.3) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain-of-Thought drops 72.1 -> 57.2 and RULER-CWE 83.7 -> 52.3 as junk ratio rises from 0% to 100%. Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion in reasoning: models increasingly truncate or skip chains. Second, partial but incomplete healing is observed: scaling instruction tuning and clean continual pre-training improve the declined cognition, yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that social effects of data could be a causal driver of LLM capability decay in continual pre-training, thereby motivating routine "cognitive health checks" for deployed and evolving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19331v1">LLM-jp-4 Technical Report</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Results for the LLM-jp-4 33B dense model are not included in this version and will be added in a future revision
    </div>
    <details class="paper-abstract">
      This paper presents LLM-jp-4, a family of Japanese sovereign large language models (LLMs) developed with an emphasis on transparency, reproducibility, and practical usability. LLM-jp-4 comprises an 8B dense model and a 32B-A3B mixture-of-experts model, both trained from scratch on approximately 12 trillion tokens. To support independent verification and reproduction, we establish explicit criteria for selecting training data, external models, and software, and make the development process publicly available. Evaluation results demonstrate substantial improvements over previous LLM-jp models and competitive performance against open-weight models of similar scale. By documenting the development principles, training procedures, and empirical findings of LLM-jp-4, we aim to provide a reference for future sovereign AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.17989v2">Whom Do AI Agents Work For? Role Assignment Induces Sponsorship Bias in LLM Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now serve as conversational shopping assistants on platforms that also sell advertising. These AI agents face a conflict of duty. They advise consumers who rely on their judgment, yet are deployed by platforms that benefit when sponsored listings are chosen. Sponsorship disclosures, designed to allow consumers to penalize paid placements, now reach the AI agent rather than the consumer, and the agent's evaluation of them is hidden from the consumer. Drawing on the fiduciary concept of conflict of duty, we argue that an agent's evaluation of a sponsored listing should not depend on which party deployed it. In controlled choice experiments, we manipulate assigned roles in the system prompt to name either a traveler or a booking platform as the agent's principal. Platform delegation significantly attenuates the penalty that agents apply to sponsored listings and weakens the skepticism that disclosure triggers in their reasoning traces. We replicate out findings across LLMs and reasoning depths. A second study decomposes the disclosure label and shows that the divergence between the two delegates widens significantly when the paid placement is attributed to the platform. Stricter terminology ("Sponsored" instead of "Promoted") lowers choice of paid listings but does not close this gap when the platform is named. The findings show that disclosure mandates designed for human consumers cannot by themselves protect consumers in AI-mediated commerce.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19314v1">PersEvalBench: Can LLM Evaluators Judge Contextually Appropriate Use of Persona Information?</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) evaluators are increasingly used to assess personalized assistants, yet whether they can reliably judge personalization has rarely been tested. We introduce PersEvalBench, a benchmark testing whether an evaluator can recognize appropriate and inappropriate uses of user history. It contains 403 multi-episode scenarios where user information should persist, transfer, or cease to apply. The benchmark covers nine controlled error subtypes across three families: under-personalization, over-personalization, and misapplied-state errors. Human audits confirm that generated errors match their intended types and that the prescribed uses of user information are reasonable. Across five LLM evaluators, recognition of controlled personalization errors remains limited even with the full dialogue history. On the human-evaluated subset, under-personalization shows the largest gap: humans recognize 94.4% of these errors, compared with 63.9% for the strongest reported evaluator. Evaluators often recognize that a response is problematic but reverse the error direction, mistaking failure to follow an applicable preference (UNDER) for use of an inapplicable preference (OVER). Easier evidence access offers little help: highlighting the relevant history turns raises error recognition only slightly, suggesting that the remaining difficulty lies in judging what role that information should play in the response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19306v1">Did the LLM Answer the Question? A Semantic Reconstruction Game for Jailbreak Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Jailbreak attacks against large language models (LLMs) are evaluated using attack success rate (ASR), which depends on correctly identifying responses that fulfill prohibited requests. Existing evaluators can mistake refusals or topically related responses for successful attacks. We show that changing only the judge LLM can alter reported ASR by up to $82$ percentage points. We introduce the semantic reconstruction game, a training-free evaluator that decomposes compliance judgments into constrained semantic comparisons. Given only a response, an LLM reconstructs the request answered by its substantive content, excluding information merely repeated or refused. The game compares this reconstruction against similar decoy requests to determine whether it matches the original request, allowing abstention when neither candidate matches. Across three off-the-shelf LLM judges, the game achieves the lowest mean error against human labels among evaluated methods and reduces variation in reported ASR. It prioritizes avoiding false positives, even at the cost of missing some successful jailbreaks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19291v1">Can LLM Agents Efficiently Identify Appliance Faults? Insights From Diagnosing a Simulated Appliance</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      In so-called "Industry 4.0," integrated sensors, analytics tools, and other digital technologies supplement industrial workflows, potentiating enhanced visibility and automation. Prognostics and health management (PHM) and natural language processing (NLP) practitioners alike have explored the application of large language models (LLMs) to industrial asset management, but many prior works' focus on exam-like benchmarks has limited their assessment of LLMs' ability to automate practical operations. Herein, we introduce CoffeeBench, a fault diagnostics task for LLM agents, using a simulated appliance. The LLMs we evaluate struggle to consistently identify faults from the sensor information alone, but the best-performing ones can iteratively refine the fault-option space to eventually obtain a successful diagnosis. Our subsequent analyses, therefore, focus on: differentiating models by diagnostic efficiency (RQ1); examining the lagging performance of the three worst models (RQ2); and identifying viable heuristics for selective prediction (RQ3), including that initial diagnostic uncertainty and LLM-judged soundness of reasoning are high-signal predictors of diagnostic success. In addition to our benchmark results (Section 3.1), we provide practical insights for improving the performance of the open-weight models on CoffeeBench, such as by explicitly tracking episode state on their behalf to reduce timeouts (Section 3.2) or applying a deliberation loop to refine diagnostic hypotheses (Section 3.3).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19267v1">MemeTrace: A Multimodal LLM-Based Agent for Safety-Aware Meme Provenance Analysis</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Online memes are often carriers of unsafe content such as disinformation and propaganda. Their rapid evolution in response to news events and strong dependence on contextual information make it difficult for platforms and moderators to detect and moderate them effectively. Moreover, existing moderation studies typically classify memes as safe or unsafe based on individual images, lacking the ability to perform broader provenance analysis, i.e., where a meme originates, how it spreads, what cultural symbolism it represents, whether it poses potential risks to society, etc. To bridge this gap, we introduce MemeTrace, a multimodal LLM-based agent that automatically traces the provenance and potential risks of emerging memes. We evaluate MemeTrace on Meme100, a dataset of 100 real-world memes, using both human annotation and automatic metrics. The results demonstrate that MemeTrace substantially outperforms the VLM baseline. More importantly, MemeTrace achieves higher accuracy in detecting unsafe memes, especially those involving disinformation and sensitive content, and remains robust under visual perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13220v2">LLM-as-an-Investigator: Evidence-First Agentic Reasoning for Robust Interactive Problem Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used for technical troubleshooting, where problem descriptions are often incomplete and multiple causes can explain the same symptoms. Effective diagnosis requires acquiring discriminative evidence and selecting the explanation best supported by it. Yet reasoning or interaction alone may not prevent premature commitment to an incorrect hypothesis. We introduce LLM-as-an-Investigator, an evidence-first methodology for interactive problem diagnosis that reduces premature commitment while distinguishing competing causes. We implement it through the Solution Investigator Agent (SIA), which orchestrates three complementary reasoning behaviours through a shared investigation state: a Diagnostic Reasoner that conducts and reassesses the diagnostic interview, a Hypothesis Investigator that maintains and updates competing hypotheses, and a Counterfactual Investigator that challenges current explanations and guides discriminative follow-up questions. Final selection uses cited dialogue evidence and a separate verification step, while explicit hypothesis and evidence records make the diagnosis traceable and preserve unresolved alternatives. We construct a structured corpus from 303 technical forum threads across mechanical, electrical, and hydraulic domains. A source-grounded pipeline separates the initial problem, pre-solution clarifications, and reported outcome, while a user simulator answers diagnostic questions without access to the reference solution. We compare SIA with direct, reasoning-based, interactive, and search-based baselines, including adaptations of ReAct and Tree of Thoughts. Evaluation distinguishes final diagnostic quality from hypothesis coverage and oracle potential, while ablations and paired stress tests assess component contributions and robustness to plausible but misleading user hypotheses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19240v1">FinGREAT: Evaluating LLMs on Financial Graph-guided Retrieval And Typing</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Financial disclosures express the underlying relationships of double-entry accounting through unstructured text. While financial LLM benchmarks predominantly emphasize question answering and classification, schema-guided evidence discovery remains relatively under-explored. We investigate whether open-weight LLMs can extract and classify evidence corresponding to predefined double-entry accounting relationships underlying the text. Using a financial accounting graph comprising 11 nodes (accounts) and 11 edges (transaction types), we evaluate mid-sized models from 5 open-weight families across 1,300 SEC 10-K Item 7 filings spanning 26 years, comparing their extraction performance against three proprietary models. Results reveal substantial differences in extraction volume, edge distributions, and cross-model agreement. Building on this study, we introduce a human-annotated relation classification benchmark comprising 483 passages across three financial accounting categories, evaluating 34 open-weight and 10 proprietary models. The strongest open-weight and proprietary models achieve 84.1\% and 92.8\% accuracy, respectively, with considerable variation across accounting categories. Our findings highlight the importance of evaluating evidence discovery and relation understanding separately, while investigating locally deployable models for scalable, privacy-sensitive financial analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19238v1">LLMs program using Reusable and Composable Abstract Representations</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      When writing code, the same computation can be expressed in different programming languages and through different implementation strategies. We show that the activations of LLMs capture this shared structure in reusable and composable abstract representations when they generate code. These representations emerge in the middle layers, are reusable across programming languages, and afford the model an impressive amount of generalization ability when generating code under various interventions. Using activation steering, we show that function, programming language, and implementation strategy can be independently manipulated. The independent interventions on language and function can be flexibly composed to control code generation, even in cases when the prompt contains no information about language and function. We additionally find that finetuning on examples of arbitrarily defined functions in one programming language transfers to other languages, consistent with the models reusing abstract features of code generation that are shared across different examples. Finally, to study how such abstract representations come to be, we implement a minimal model of shared, reusable, and composable structure in a synthetic setting, and show that transformers trained on such data learn representations that reuse these parts flexibly across contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19206v1">DUALVIEW: Leveraging Broad and Output-Specific Evidence for LLM-based Multi-Output Text Regression</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      While work on LLM-based regression has largely focused on scalar prediction, multi-output text regression remains less explored. We present DualView, a parameter-efficient approach to leverage LLMs for multi-output text regression. \method{} first extracts broad and output-specific representations from a frozen LLM, then uses a dual-branch architecture to produce the full score vector. Across five heterogeneous regression tasks and eight LLMs (up to 122B), DualView outperforms seven supervised regression baselines while requiring 82--1842 times fewer fitted parameters than the three strongest baselines. These improvements extend across individual output dimensions and persist even with limited training data. Notably, DualView using a smaller open-weight LLM can outperform unsupervised prediction with Gemini 3.1 Pro. Controlled ablations show that broad and output-specific evidence provide complementary predictive benefits, while lightweight shared predictors achieve strong performance with fewer trainable parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.27296v3">LLMs Can Design Near-Optimal OR Algorithms</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      We ask whether large language models (LLMs) can design effective algorithms for well-specified operations research (OR) problems. We study inventory control, queueing network control, and assortment optimization, using benchmark instances from recent papers in each area. We evaluate two levels of LLM use: at level 1, the model receives one problem instance and returns a solution for that instance; at level 2, it receives only the problem class description and broad parameter ranges, and returns an algorithm that maps instance parameters to solutions. Human input is minimal: we give one untuned prompt that describes the problem, and the model has access to a Python sandbox tool with a fixed compute budget. The strongest model we test, gpt-5.6-sol, matches or outperforms the best method reported in those papers on almost all evaluated instances. This holds even at level 2, where the returned algorithm is fixed before seeing the evaluation instances. Performance also improves sharply across models released less than eight months apart, suggesting that this capability is moving quickly. Thus, for the well-specified operations problems we study, a single untuned LLM query can already produce algorithms competitive with specialized methods. These results suggest that frontier LLMs can be a serious empirical baseline for algorithm design in well-specified OR problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19198v1">Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 50 pages, 7 figure, 45 tables
    </div>
    <details class="paper-abstract">
      With the growing number of LLMs available, selecting which one to use for a prompt has become a key question. A large part of the routing literature relies on semantically embedding prompts into a vector space and then applying a lightweight classifier to the embedded representations. These methods perform robustly in-distribution but they struggle to generalize to unseen tasks. By studying the geometry of semantic embeddings, we trace a clear failure mode: features that determine the neighbors of unseen prompts are an unreliable proxy for LLM selection. We therefore replace the semantic embedding with the following map: we run a partial forward pass of Qwen2.5-7B-Instruct on the prompt and use the hidden state of its last token at a deep layer as the representation. When applying kNN and other simple classifiers to this new representation, we obtain a clear result in generalization: on average across cost penalties, their performance improves. Moreover, kNN and another simple classifier that we propose outperform every evaluated semantic-embedding baseline. Our two simple methods also beat a recent SoTA hidden-states based router on RouterBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11414v2">Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted at the 7th ACM International Conference on AI in Finance (ICAIF '26), Milan, Italy, November 14-17, 2026. 8 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in financial applications fail most consequentially when they are confidently wrong. Hedged, uncertain answers invite scrutiny, whereas confident errors silently degrade downstream decisions without warning. We ask how reliably such confidently wrong answers, or confident hallucinations, can be detected from a model's internal activations, and whether those activations carry information beyond its output-level confidence signals. We train linear probes on the residual stream and evaluate them on two established numerical question-answering (QA) benchmarks built from real filings, FinQA and TAT-QA. Behavioral confidence is measured as the agreement among eight resampled answers to the same question, and probe effectiveness is compared against baselines, such as token log-probabilities and the model's own True/False self-assessment of its answer. Our findings show that among confident answers, those for which all eight resamples agree, 15-23% are wrong on FinQA. There the probes have a significant advantage over output-level baselines in detecting hallucinations, holding 0.68-0.77 AUROC while the best baselines fall to 0.55-0.63, across Qwen3-8B, Llama-3.1-8B, and Gemma-2-9B. Our results suggest that probing can be a cost-effective triage mechanism for routing LLM answers to human review and quality control procedures in high-stakes financial applications. Code and data are available at https://github.com/richardzhewang/confidently-wrong.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.09015v2">Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      We study contextual bandit problems with correlated arms and access to surrogate reward signals produced by a machine learning model, motivated by applications such as large language model (LLM) routing. Unlike classical contextual bandits that rely solely on bandit feedback and assume conditional independence across arms, our setting allows context-dependent inter-arm correlations and auxiliary reward information that may be noisy or misspecified. We propose algorithms that leverage such surrogate rewards through two complementary designs. A coupled reward-mixing approach pools true and surrogate rewards to accelerate learning when surrogate signals are reliable, while a decoupled prediction-mixing approach maintains separate estimators for bandit feedback and surrogate rewards and adaptively combines their predictions. This decoupling yields robustness to surrogate misspecification, recovering regret guarantees comparable to reward-only bandit methods in the worst case, while achieving improved regret when surrogate predictions are sufficiently informative. We provide theoretical regret analyses for both approaches and evaluate them on LLM routing benchmarks under varying accuracy versus cost trade-offs. The results demonstrate improved sample efficiency and consistently better accuracy-cost trade-offs compared to standard contextual bandit baselines and strong static routing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19167v1">Beyond Functional Correctness: Static Code Quality Profiling of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted for publication in ICECCS2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for code generation, yet most evaluations still focus on whether generated code passes benchmark test suites. Functional correctness alone, however, provides only a limited view of quality: benchmark tests are often incomplete, and passing them does not guarantee that code is maintainable, reliable, or well-structured. This paper {presents a large-scale static quality profiling of} LLM-generated Python code, with a focus on maintainability and reliability. Using 65{,}663 programs from 73 models in LiveCodeBench, we evaluate generated code with established static analysis tools, including Pylint, Radon, and SonarQube. Our results show systematic differences across models that are not captured by correctness alone, with maintainability emerging as the clearest source of variation. We further find that correctness often comes with structural trade-offs rather than uniformly simpler solutions. Overall, our findings motivate evaluation frameworks that assess LLM-generated code beyond pass rates alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19158v1">AgentCIScope: Grounded Contextual Integrity Evaluation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      LLM agents are becoming increasingly capable of handling daily workspace tasks by interacting with application environments through tool calls. However, as agents read and write sensitive user data across email, documents, and other applications, a central privacy question is whether they preserve contextual integrity (CI), that is, whether the information flows they produce comply with context-specific disclosure constraints. Existing work either focuses on model-level privacy leakage or evaluates agents in limited application environments with manually constructed disclosure constraints. In this work, we introduce AgentCIScope, a unified platform and benchmark for evaluating agent-level contextual integrity through executable, multi-step tasks across diverse, realistic application environments. AgentCIScope transforms public and synthetic privacy cases into executable agent tasks, with disclosure constraints grounded in three criteria: governing privacy policies, persona-based agreement, and a predefined sensitive-attribute taxonomy. Based on AgentCIScope, we evaluate 12 agents on 936 tasks and find that strong task-completion capability does not guarantee contextual-integrity preservation: single-run leakage rates range from 32.5% to 94.4%, while repeating each task five times further increases leakage by 4-19 percentage points. Leakage varies substantially across attributes: personal preferences and booking details have average leakage rates of 84.7% and 76.5%, compared with 17.1% for credit-card information and 29.4% for SSNs; environments: calendar has the highest average leakage rate as an information source, while WhatsApp ranks highest as an information destination; and governing privacy policies. AgentCIScope provides deep insights into agent privacy leakage and serves as a unified testbed for developing privacy-preserving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19062v1">RLPE: Teaching LLMs to Prompt with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 25 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Prompting is a widely used method for adapting general-purpose LLMs to domain-specific tasks. Effective prompts can be tedious to create manually, so prompt writing is increasingly handed to LLMs, with recent work training LLMs to write prompts for other models. These techniques either write prompts without observing how the task model behaves, blind to the failure modes the prompt should address, or must run alongside the task LLM at inference time, adding overhead to every query. Neither reflects how prompt engineering is often practiced. To close this gap, we introduce Reinforcement Learned Prompt Engineering (RLPE), which trains a model to analyze traces of a separate task model and write a prompt targeting the failures it observes. RLPE rewards the trained model strictly on prompt performance, so it learns the prompting strategies that improve the task model. The prompt is generated once and reused across the task, so inference does not require the trained model. Since RLPE needs only query access to the model it prompts, it is also compatible with closed-source models. We apply RLPE to Enigmata, BIG-Bench Extra Hard, and PlanBench. We train a 9B-parameter model that writes better prompts than state-of-the-art frontier models (gpt-5.6-sol and claude-opus-5) and prompt optimizers (GEPA and TextGrad) on 2/3 benchmarks. Although the model is never explicitly trained to iterate on prompts, this capability emerges at a level that matches or beats frontier models. The learned prompting capability also generalizes, with the trained model improving task models 6x larger than the one it trained with, including ones from entirely different model families, and matching claude-opus-5 on a fourth benchmark held out from training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08552v4">Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Descriptive scientific metadata in public repositories are often incomplete and inconsistent with community standards and ontologies, limiting data FAIRness. Large language models (LLMs) offer a promising approach to automatically standardizing such metadata when the LLM is provided with knowledge of the relevant standards in machine-actionable form (e.g., as metadata templates from systems such as the CEDAR Workbench). Entry of metadata standards through prompt engineering offers only fixed snapshots of the relevant standards, however, and it relies on an LLM's pretrained knowledge to interpret and satisfy constraints related to the standard. We evaluate whether giving an LLM access to metadata specifications and authoritative terminology at runtime improves automated metadata standardization. We present ARMS, a tool-augmented LLM agent that retrieves complete CEDAR metadata templates and dynamically queries authoritative biomedical terminology services at execution time. We compared ARMS with a prompt-based approach on 839 legacy metadata records from the Human BioMolecular Atlas Program (HuBMAP), using expert-standardized records as the reference standard. ARMS outperformed the prompt-based approach, increasing precision from 0.56 to 0.93 and recall from 0.51 to 0.85, with improvements across all field categories and assay types. The largest gains occurred for ontology-constrained fields, where precision increased from 0.36 to 0.92. Our findings show that access to appropriate metadata standards can substantially improve LLM-based standardization of legacy metadata. ARMS improves metadata standardization by giving LLMs runtime access to the authoritative resources that define valid metadata. Representation of metadata standards in machine-actionable form enhances the task of rectifying legacy metadata using LLM technology, especially when those standards can be queried dynamically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20645v4">Anticipatory Evaluation of Language Models: Predicting LLM Benchmark Scores from Descriptions</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 29 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Modern ML research increasingly faces an evaluation bottleneck: building benchmarks and running experiments is costly, slowing iteration. We ask how much can be inferred about evaluation outcomes before experiments, given only a natural-language description of the task and setup. We call this setting anticipatory evaluation. We introduce EVALCAST, a literature-grounded resource of 2,290 description-performance pairs from 1,519 arXiv papers, including a test set of papers published after the model's knowledge cutoff. We evaluate two tasks: predicting absolute performance through regression and comparing benchmark difficulty through pairwise prediction. Experiments with frontier LLMs show that accurate regression remains challenging, while pairwise predictions can recover relative performance ordering under matched settings. Comparable performance on post-cutoff and streaming evaluations suggests that the observed signal is not solely attributable to previously reported results. Overall, EVALCAST provides a resource and evaluation framework for studying anticipatory evaluation and its potential use in prioritizing benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19021v1">Can LLMs Access their World Knowledge for Event Prediction?</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Recent work has questioned whether large language models (LLMs) can be useful for temporal event prediction. Despite vast pretrained world knowledge, LLMs often underperform simpler specialized models. We ask whether this knowledge is truly inaccessible, or whether current methods fail to present event prediction in a form LLMs can use. We argue for the latter and show that LLMs become effective event forecasters when the interface is designed around two choices. The first grounds each event in a natural-language description, and the second separates event semantics from the prediction space to prevent semantic bias from label associations. To test this, we introduce DescribeEvent, a benchmark of ~150K events across seven datasets and six domains with event-level descriptions. We find that capable LLMs substantially benefit from descriptions and symbolic labels, outperforming description-free variants and non-LLM baselines, while weaker models gain little. These results suggest that accessing LLM world knowledge for event prediction is an interface problem, and that event descriptions offer a practical way to elicit the knowledge LLMs already encode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.18857v1">PixelMoE: Scaling Pixel-Space Diffusion with LLM-Style Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Pixel-space diffusion models provide an appealing alternative to latent-space diffusion models by directly modeling raw images without relying on a learned tokenizer. However, scaling pixel-space models has predominantly followed dense strategies, tightly coupling model capacity with active computation. In this work, we show that standard LLM-style Mixture-of-Experts (MoE) architectures can be effectively transferred to pixel-space diffusion. Building on this finding, we introduce PixelMoE, a pixel-space generative framework that sparsely scales model capacity with standard LLM-style MoE, without diffusion-specific routing mechanisms or auxiliary routing objectives. Specifically, PixelMoE combines standard Token-Choice (TC) routing with fine-grained routed experts for input-adaptive sparse computation, a shared expert for capturing common representations, and a bias-based load-balancing strategy. To complement sparse capacity scaling, PixelMoE further introduces Pixel-Space Hierarchical Flow Matching, which varies spatial resolution along the denoising trajectory from global structure formation to fine-grained detail refinement. At comparable active parameter counts, PixelMoE consistently outperforms its dense counterparts across model scales, demonstrating effective sparse capacity scaling in pixel space. On class-conditional ImageNet-1K, PixelMoE achieves FID scores of 1.47 at 256x256 and 1.53 at 512x512 without representation alignment or perceptual losses, comparing favorably with leading pixel-space and latent-space generative models. Code and models will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.18762v1">Looking Complete Is Not Being Grounded: Evaluating LLM Sitreps Against Human-Authored Situation Reports</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted at LT4CPR Workshop @ AACL-IJCNLP 2026
    </div>
    <details class="paper-abstract">
      Evaluation for automatic crisis reporting should measure what practitioners need, not what is convenient to compare. Through LLM-assisted, author-approved content analysis, we derive a 15-slot information schema from professional humanitarian situation reports using qualitative content analysis, then evaluate 121 human and 261 LLM-generated reports across five disasters and three hazard types. Under google/gemini-3.7-flash on Cyclone Idai, the best machine arm is nominally above the empirical human ceiling (0.827 vs. 0.821), but the difference is statistically indistinguishable from zero (95\% bootstrap CI $[-.049,.059]$; permutation $p=.852$), and the ordering does not replicate under openai/gpt-5.6-luna. On a blind 100-item Idai sample, the author agrees with the Gemini judge at 0.910 accuracy and unweighted Cohen's $κ=.799$. All four judges clear the plan's raw-agreement criterion of .7; only Gemini reaches .7 on chance-corrected $κ$. Across the three primary events, machine reports reproduce 5.4-11.0% of same-day official figures. Across those same events, raw exact-number matching links 14.3-19.6% of official figures and 85.4-90.1% of machine figures to the social stream; manual audit confirms 45/50 declared absences, 4/25 official matches, and 24/25 machine matches semantically. The issuing-provenance gap is positive in all 12 observed event-judge cells: report date, organisation, series, and authority were not supplied at professional-report coverage in any cell. On the three primary events, ROUGE-L is uninformative, BERTScore is negatively associated with completeness after multiplicity correction, and a generic LLM judge changes sign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30391v2">Energy-Aware Scheduling for Serverless LLM Serving on Shared GPUs</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 13 pages body and 4 pages appendix, 19 pages total. Accepted to the 2026 ACM SIGOPS Annual Technical Conference (SIGOPS ATC 2026)
    </div>
    <details class="paper-abstract">
      As LLM inference becomes a major cloud workload, its growing energy footprint makes cluster-wide energy optimization increasingly important. Serverless LLM serving helps platforms absorb traffic volatility by elastically sharing GPU resources across models, but this sharing also makes energy optimization difficult. Multiple co-resident models run under one device-wide operating point, while their resource demands and latency slack change across execution phases and load conditions. As a result, minimizing energy requires coordinated scheduling across request placement, runtime resource adaptation, and workload consolidation. We present Festina, a profiling-guided, power-aware control plane to minimize cluster-wide energy for serverless LLM serving. Unlike common global-local schedulers that focus on throughput or tail latency, Festina makes energy-first decisions by jointly coordinating request placement, SM partitioning, and GPU operating points under TTFT/TBT SLOs. In our system, a lightweight global scheduler performs fast, SLO-safe, energy-aware placement using constant-time lookups from offline profiles and GPU state summaries. On each GPU, a phase-aware local scheduler continuously adapts task batching and compute resources to minimize power consumption. Festina further performs energy-aware workload consolidation to reduce GPUs' static power consumption via SLO-aware migration. Comparison with four SOTA LLM serving systems and one DVFS-augmented system demonstrates that Festina reduces energy consumption by up to 56% while maintaining parity in SLO attainment (within a 2% margin)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19225v6">FinTradeBench: A Financial Reasoning Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 8 pages main text, 36 pages total (including references and appendix). 9 figures, 31 tables. Preprint under review. Code and data will be made available upon publication
    </div>
    <details class="paper-abstract">
      Financial analysis requires reasoning over heterogeneous signals, including company fundamentals from regulatory filings and trading signals derived from market prices. However, existing financial benchmarks focus almost exclusively on regulatory filings, neglecting market dynamics and cross-signal interactions. We introduce FinTradeBench, a benchmark of 1,400 questions covering NASDAQ-100 stocks over a ten-year window. To holistically evaluate financial reasoning, FinTradeBench includes questions in three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions that require cross-signal reasoning. Our benchmark starts with 150 expert-authored seed questions, which are then scaled to 1400. To ensure reliable evaluation, we create a Core400 subset in which every item is expert-reviewed. In zero-shot (No-RAG) response generation, we observe that all models perform substantially worse on trading-focused questions compared to fundamental-focused questions. To identify failure causes, we evaluate LLMs under controlled settings including retrieval-based baselines, tool-assisted computation, oracle conditions, and time-series-specialized pipelines. LLMs' capabilities to do computation and evidence selection contribute most to reasoning failures. Additionally, retrieval significantly improves trading-signal reasoning but does not improve fundamental reasoning, where identifying relevant signals remains the dominant failure mode. Finally, rather than relying on a small, fine-tuned model for time-series reasoning, it is better to summarize the time-series data and leverage LLMs' reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02076v3">LLM-as-Jev: LLMs Are Already Jev-Style Decision Models -- When and How to Fine-Tune Them</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Jev-style decision models have gained surging traction by making decision-making dramatically faster and more efficient, returning categorical probability distributions over predefined options without generating free-form text. However, because Jev's architecture and training methodologies remain proprietary, how to replicate these capabilities effectively remains an open question. In this work, we investigate the extent to which general-purpose LLMs already possess this capability out of the box, and when fine-tuning is actually necessary. We present LLM-as-Jev, an architecture-preserving framework offering both a training-free inference recipe and a fine-tuning objective with KL-divergence anchors. Across Qwen3.5-4B and Qwen3-0.6B, we find that modern LLMs are inherently effective decision models: off the shelf, frozen models match or exceed existing training-free readouts on the same 4B backbone, scale to arbitrary option counts, and natively handle multimodal decisions. They also seamlessly inherit the backbone's thinking mode: while Jev was introduced as a System 1 model, "Thinking Jev" leverages budgeted test-time reasoning to boost accuracy on challenging tasks without training. Fine-tuning provides targeted gains for smaller models and complex routing, while our KL anchors prevent behavioral degradation in general text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03762v2">OracleProto: A Reproducible Framework for Benchmarking LLM Forecasting via Knowledge Cutoffs and Temporal Masking</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Forecasting is a composite capability encompassing information gathering, evidence integration, situational assessment, and decision-making. Large language models (LLMs) combine reasoning and tool use, offering new avenues for automated forecasting and decision support. Yet LLM forecasting evaluation faces a structural challenge: live benchmarks support prospective assessment, but limited question reuse and changing information conditions make results difficult to reproduce and compare; retrospective benchmarks offer reproducibility but risk information leakage. We introduce OracleProto, a reproducible framework for evaluating LLM forecasting in a controlled, temporally masked environment. Through model knowledge cutoff constraints and temporal masking, OracleProto constructs a temporal sandbox for real-world web retrieval, audits information boundaries through leakage detection, and reconstructs resolved events as forecasting instances. Discrete answer normalization and hierarchical scoring enable unified evaluation. We evaluate 27 model-configuration pairs on 300 FutureX-Past-derived questions over 24,300 prediction trials, assessing forecasting quality, sampling stability, and cost efficiency under controlled information boundaries. An audit finds only 1.37% residual leakage in retrieval results after temporal masking and leakage filtering. OracleProto retains resolved forecasting tasks as reusable evaluation resources under explicit temporal boundaries, supporting continual accumulation of forecasting datasets and systematic evaluation of forecasting capabilities. Code is available at https://anonymous.4open.science/r/OracleProto-2E3A/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15020v3">What Users See Is Not What Models Read: Split-View PDFs in Document-to-LLM Supply Chains</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted by IEEE Symposium on Security and Privacy (S&P 2027)
    </div>
    <details class="paper-abstract">
      Document-to-LLM applications typically read uploaded PDFs by first translating them into text through an extraction layer whose output is not ordinarily exposed to users. We show that this layer enables split-view PDFs: one document can have two semantic views before model reasoning. By mining specification-permitted or implementation-tolerated representation gaps at the PDF render/extract boundary, we instantiate 25 extraction gaps (EG) in which extractors return attacker-controlled or extractor-dependent text while the rendered page shows benign or different content. The gaps form four families: semantic overrides, hidden semantic injection, reading-order splits, and font-decoding splits, and 14 gaps have no exact path/mechanism-level match in prior PDF-to-LLM attacks. We evaluate these gaps on 16 PDF processing stacks and 7 commercial LLM services. Each gap causes render-extract divergence on at least one stack. Under a gap-level exposure criterion, every evaluated service exposes at least one gap, with 12/25 to 21/25 exposed gaps. Exposure is driven mainly by the ingestion stack -- not model identity alone. We further show that tested safety filters cover only selected hidden-text constructions. To support triage, we develop a static screening scanner whose rules trigger on all 25 benchmark gaps, and discuss dual-view consistency as a longer-term defense direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12390v1">Long Text to Predictive Features: LLM-Guided Blockwise Feature Engineering via Executable Program Search</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Industrial risk-control systems typically rely on structured-data models for efficient prediction, yet substantial valuable information remains embedded in unstructured long text. Extracting this information through manual feature engineering is labor-intensive, while requiring a large language model (LLM) to process every real-time input may not meet practical deployment requirements. To address this challenge, we propose LLM-BlockFE, an LLM-guided offline feature construction framework that converts long text into executable feature programs, thereby avoiding LLM calls during online inference. LLM-BlockFE constructs feature programs by incrementally appending immutable code blocks and evaluates candidate features using a downstream model. To address the tendency of conventional greedy search to become trapped in suboptimal solutions, our method introduces a block-level rollback mechanism based on depth-calibrated credit allocation and advances multiple independent search trajectories in an interleaved manner, reducing redundant exploration by sharing fixed descriptions of each trajectory's exploration direction. After the search, the resulting programs are frozen and deployed to extract structured features for downstream prediction models. Across two public and two private datasets, LLM-BlockFE achieves absolute AUC improvements of 0.0069 to 0.0358 over the strongest baseline on each dataset in the full-dataset comparison. Post-launch monitoring across five deployed financial risk-control applications shows absolute KS improvements of 0.02 to 1.56 percentage points over the existing manually designed strategy.
    </details>
</div>
