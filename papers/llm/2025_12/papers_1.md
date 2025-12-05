# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05105v1">Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Long context reasoning in large language models (LLMs) has demonstrated enhancement of their cognitive capabilities via chain-of-thought (CoT) inference. Training such models is usually done via reinforcement learning with verifiable rewards (RLVR) in reasoning based problems, like math and programming. However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency. As a result, it requires significant compute resources in post-training phase. To overcome these limitations, in this work, we propose \textbf{Semantic Soft Bootstrapping (SSB)}, a self-distillation technique, in which the same base language model plays the role of both teacher and student, but receives different semantic contexts about the correctness of its outcome at training time. The model is first prompted with a math problem and several rollouts are generated. From them, the correct and most common incorrect response are filtered, and then provided to the model in context to produce a more robust, step-by-step explanation with a verified final answer. This pipeline automatically curates a paired teacher-student training set from raw problem-answer data, without any human intervention. This generation process also produces a sequence of logits, which is what the student model tries to match in the training phase just from the bare question alone. In our experiment, Qwen2.5-3B-Instruct on GSM8K dataset via parameter-efficient fine-tuning. We then tested its accuracy on MATH500, and AIME2024 benchmarks. Our experiments show a jump of 10.6%, and 10% improvements in accuracy, respectively, over group relative policy optimization (GRPO), which is a commonly used RLVR algorithm. Our code is available at https://github.com/purbeshmitra/semantic-soft-bootstrapping, and the model, curated dataset is available at https://huggingface.co/purbeshmitra/semantic-soft-bootstrapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05066v1">Multi-LLM Collaboration for Medication Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 8 pages, 5 figures, 1 table
    </div>
    <details class="paper-abstract">
      As healthcare increasingly turns to AI for scalable and trustworthy clinical decision support, ensuring reliability in model reasoning remains a critical challenge. Individual large language models (LLMs) are susceptible to hallucinations and inconsistency, whereas naive ensembles of models often fail to deliver stable and credible recommendations. Building on our previous work on LLM Chemistry, which quantifies the collaborative compatibility among LLMs, we apply this framework to improve the reliability in medication recommendation from brief clinical vignettes. Our approach leverages multi-LLM collaboration guided by Chemistry-inspired interaction modeling, enabling ensembles that are effective (exploiting complementary strengths), stable (producing consistent quality), and calibrated (minimizing interference and error amplification). We evaluate our Chemistry-based Multi-LLM collaboration strategy on real-world clinical scenarios to investigate whether such interaction-aware ensembles can generate credible, patient-specific medication recommendations. Preliminary results are encouraging, suggesting that LLM Chemistry-guided collaboration may offer a promising path toward reliable and trustworthy AI assistants in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08123v5">QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2025, pages 20619-20642, Suzhou, China
    </div>
    <details class="paper-abstract">
      Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00030v3">SignBind-LLM: Multi-Stage Modality Fusion for Sign Language Translation</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Despite progress in gloss-free Sign Language Translation (SLT), traditional single modality end-to-end approaches consistently fail on two critical components of natural signing: the precise recognition of high-speed fingerspelling and the integration of asynchronous non-manual cues from the face. Recent progress in SLT with Large Language Models has side stepped this challenge, forcing a single network to learn these simultaneously resulting in poor performance when tasked with translating crucial information such as names, places, and technical terms. We introduce SignBind-LLM, a modular framework designed to overcome these limitations. Our approach employs separate, specialized predictors for continuous signing, fingerspelling, and lipreading. Each expert network first decodes its specific modality into a sequence of tokens. These parallel streams are then fused by a lightweight transformer that resolves temporal misalignments before passing the combined representation to a Large Language Model (LLM) for final sentence generation. Our method establishes a new state-of-the-art on the How2Sign, ChicagoFSWildPlus, and BOBSL datasets with a BLEU-4 score of 22.1, 73.2% letter accuracy and BLEU-4 score of 6.8 respectively. These results validate our core hypothesis: isolating and solving distinct recognition tasks before fusion provides a more powerful and effective pathway to robust, high-fidelity sign language translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04957v1">LLMs Know More Than Words: A Genre Study with Syntax, Metaphor & Phonetics</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable potential across diverse language related tasks, yet whether they capture deeper linguistic properties, such as syntactic structure, phonetic cues, and metrical patterns from raw text remains unclear. To analysis whether LLMs can learn these features effectively and apply them to important nature language related tasks, we introduce a novel multilingual genre classification dataset derived from Project Gutenberg, a large-scale digital library offering free access to thousands of public domain literary works, comprising thousands of sentences per binary task (poetry vs. novel;drama vs. poetry;drama vs. novel) in six languages (English, French, German, Italian, Spanish, and Portuguese). We augment each with three explicit linguistic feature sets (syntactic tree structures, metaphor counts, and phonetic metrics) to evaluate their impact on classification performance. Experiments demonstrate that although LLM classifiers can learn latent linguistic structures either from raw text or from explicitly provided features, different features contribute unevenly across tasks, which underscores the importance of incorporating more complex linguistic signals during model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04852v1">Ask Safely: Privacy-Aware LLM Query Generation for Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to query knowledge graphs (KGs) due to their strong semantic understanding and extrapolation capabilities compared to traditional approaches. However, these methods cannot be applied when the KG contains sensitive data and the user lacks the resources to deploy a local generative LLM. To address this issue, we propose a privacy-aware query generation approach for KGs. Our method identifies sensitive information in the graph based on its structure and omits such values before requesting the LLM to translate natural language questions into Cypher queries. Experimental results show that our approach preserves the quality of the generated queries while preventing sensitive data from being transmitted to third-party services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04844v1">Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Expanding the linguistic diversity of instruct large language models (LLMs) is crucial for global accessibility but is often hindered by the reliance on costly specialized target language labeled data and catastrophic forgetting during adaptation. We tackle this challenge under a realistic, low-resource constraint: adapting instruct LLMs using only unlabeled target language data. We introduce Source-Shielded Updates (SSU), a selective parameter update strategy that proactively preserves source knowledge. Using a small set of source data and a parameter importance scoring method, SSU identifies parameters critical to maintaining source abilities. It then applies a column-wise freezing strategy to protect these parameters before adaptation. Experiments across five typologically diverse languages and 7B and 13B models demonstrate that SSU successfully mitigates catastrophic forgetting. It reduces performance degradation on monolingual source tasks to just 3.4% (7B) and 2.8% (13B) on average, a stark contrast to the 20.3% and 22.3% from full fine-tuning. SSU also achieves target-language performance highly competitive with full fine-tuning, outperforming it on all benchmarks for 7B models and the majority for 13B models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04834v1">Are LLMs Truly Multilingual? Exploring Zero-Shot Multilingual Capability of LLMs for Information Retrieval: An Italian Healthcare Use Case</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become a key topic in AI and NLP, transforming sectors like healthcare, finance, education, and marketing by improving customer service, automating tasks, providing insights, improving diagnostics, and personalizing learning experiences. Information extraction from clinical records is a crucial task in digital healthcare. Although traditional NLP techniques have been used for this in the past, they often fall short due to the complexity, variability of clinical language, and high inner semantics in the free clinical text. Recently, Large Language Models (LLMs) have become a powerful tool for better understanding and generating human-like text, making them highly effective in this area. In this paper, we explore the ability of open-source multilingual LLMs to understand EHRs (Electronic Health Records) in Italian and help extract information from them in real-time. Our detailed experimental campaign on comorbidity extraction from EHR reveals that some LLMs struggle in zero-shot, on-premises settings, and others show significant variation in performance, struggling to generalize across various diseases when compared to native pattern matching and manual annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23808v3">Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      A prevailing view in Reinforcement Learning with Verifiable Rewards (RLVR) interprets recent progress through the lens of an exploration-exploitation trade-off, a perspective largely shaped by token-level metrics. We re-examine this perspective, proposing that this perceived trade-off may not be a fundamental constraint but rather an artifact of the measurement level. To investigate this, we shift the analysis to the semantically rich hidden-state space, adopting Effective Rank (ER) to quantify exploration and proposing its novel first- and second-order derivatives, named ER Velocity and ER Acceleration, to capture exploitation dynamics. Our analysis reveals that in the semantic space, exploration and exploitation could be decoupled (Sec.~4). This finding reveals an opportunity to enhance both capacities simultaneously. This insight motivates our method, Velocity-Exploiting Rank-Learning (VERL), the first to operationalize the principle of synergistic exploration-exploitation enhancement by directly shaping the RL advantage function. The key innovation is leveraging the theoretically stable ERA as a predictive meta-controller to create a synergistic, dual-channel incentive structure. Instead of forcing a trade-off, VERL prospectively amplifies rewards for exploration to preempt overconfidence and reinforces exploitative gains to consolidate reasoning. Experiments across diverse LLMs and reasoning benchmarks show consistent gains, including up to 21.4% absolute accuracy improvement on the challenging Gaokao 2024 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16275v2">SeSE: A Structural Information-Guided Uncertainty Quantification Framework for Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 14 pages of main text and 10 pages of appendices;Submit to IEEE TKDE
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding ``hallucinating'' falsehoods. However, state-of-the-art UQ methods primarily rely on semantic probability distributions or pairwise distances, overlooking latent semantic structural information that could enable more precise uncertainty estimates. This paper presents Semantic Structural Entropy (SeSE), a principled UQ framework that quantifies the inherent semantic uncertainty of LLMs from a structural information perspective for hallucination detection. SeSE operates in a zero-resource manner and is applicable to both open- and closed-source LLMs, making it an ``off-the-shelf" solution for new models and tasks. Specifically, to effectively model semantic spaces, we first develop an adaptively sparsified directed semantic graph construction algorithm that captures directional semantic dependencies while automatically pruning unnecessary connections that introduce negative interference. We then exploit latent semantic structural information through hierarchical abstraction: SeSE is defined as the structural entropy of the optimal semantic encoding tree, formalizing intrinsic uncertainty within semantic spaces after optimal compression. A higher SeSE value corresponds to greater uncertainty, indicating that LLMs are highly likely to generate hallucinations. In addition, to enhance fine-grained UQ in long-form generation, we extend SeSE to quantify the uncertainty of individual claims by modeling their random semantic interactions, providing theoretically explicable hallucination detection. Extensive experiments across 29 model-dataset combinations show that SeSE significantly outperforms advanced UQ baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04746v1">SignRoundV2: Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Extreme low-bit quantization is critical for efficiently deploying Large Language Models (LLMs), yet it often leads to severe performance degradation at 2-bits and even 4-bits (e.g., MXFP4). We present SignRoundV2, a post-training quantization framework that is highly effective even without mixed-precision. SignRoundV2 introduces (1) a fast sensitivity metric that combines gradient information with quantization-induced deviations to guide layer-wise bit allocation, and (2) a lightweight pre-tuning search for quantization scales to improve extremely low-bit quantization. These components allow SignRoundV2 to close the gap with full-precision models. Extensive experiments indicate that our method sustains competitive accuracy for LLMs, achieving production-grade performance with about 1 percent variance at 4-5 bits and strong results even at 2 bits. The implementation is available at https://github.com/intel/auto-round.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04695v1">TRINITY: An Evolved LLM Coordinator</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Combining diverse foundation models is promising, but weight-merging is limited by mismatched architectures and closed APIs. Trinity addresses this with a lightweight coordinator that orchestrates collaboration among large language models (LLMs). The coordinator, comprising a compact language model (approximately $0.6$B parameters) and a lightweight head (approximately $10$K parameters), is optimized with an evolutionary strategy for efficient and adaptive delegation. Trinity processes queries over multiple turns, where at each turn the coordinator assigns one of three roles (Thinker, Worker, or Verifier) to a selected LLM, effectively offloading complex skill acquisition from the coordinator itself. Experiments show that Trinity consistently outperforms individual models and existing methods across coding, math, reasoning, and domain knowledge tasks, and generalizes robustly to out-of-distribution tasks. On standard benchmarks, Trinity achieves state-of-the-art results, including a score of 86.2% on LiveCodeBench. Theoretical and empirical analyses identify two main factors behind this performance: (1) the coordinator's hidden-state representations provide rich contextualization of inputs, and (2) under high dimensionality and strict budget constraints, the separable Covariance Matrix Adaptation Evolution Strategy offers advantages over reinforcement learning, imitation learning, and random search by exploiting potential block-epsilon-separability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13247v3">Grounding LLM Reasoning with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at generating natural language answers, yet their outputs often remain unverifiable and difficult to trace. Knowledge Graphs (KGs) offer a complementary strength by representing entities and their relationships in structured form, providing a foundation for more reliable reasoning. We propose a novel framework that integrates LLM reasoning with KGs by linking each step of the reasoning process to graph-structured data. This grounding turns intermediate ``thoughts'' into interpretable traces that remain consistent with external knowledge. Our approach incorporates multiple reasoning strategies, Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Graph-of-Thought (GoT), and is evaluated on GRBench, a benchmark for domain-specific graph reasoning. Our experiments show state-of-the-art (SOTA) performance, with at least 26.5\% improvement over CoT baselines. Beyond accuracy, we analyze how step depth, branching structure, and model size influence reasoning quality, offering insights into the conditions that support effective reasoning. Together, these contributions highlight how grounding LLMs in structured knowledge enables both higher accuracy and greater interpretability in complex reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04668v1">Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Under review at ACL Rolling Review
    </div>
    <details class="paper-abstract">
      Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09495v2">Bridging Online Behavior and Clinical Insight: A Longitudinal LLM-based Study of Suicidality on YouTube Reveals Novel Digital Markers</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Suicide remains a leading cause of death in Western countries. As social media becomes central to daily life, digital footprints offer valuable insight into suicidal behavior. Focusing on individuals who attempted suicide while uploading videos to their channels, we investigate: How do linguistic patterns on YouTube reflect suicidal behavior, and how do these patterns align with or differ from expert knowledge? We examined linguistic changes around suicide attempts and compared individuals who attempted suicide while actively uploading to their channel with three control groups: those with prior attempts, those experiencing major life events, and matched individuals from the broader cohort. Applying complementary bottom-up, hybrid, and expert-driven approaches, we analyzed a novel longitudinal dataset of 181 suicide-attempt channels and 134 controls. In the bottom-up analysis, LLM-based topic-modeling identified 166 topics; five were linked to suicide attempts, two also showed attempt-related temporal changes (Mental Health Struggles, $OR = 1.74$; YouTube Engagement, $OR = 1.67$; $p < .01$). In the hybrid approach, clinical experts reviewed LLM-derived topics and flagged 19 as suicide-related. However, none showed significant effects beyond those identified bottom-up. YouTube Engagement, a platform-specific indicator, was not flagged, underscoring the value of bottom-up discovery. A top-down psychological assessment of suicide narratives revealed differing motivations: individuals describing prior attempts aimed to help others ($β=-1.69$, $p<.01$), whereas those attempted during the uploading period emphasized personal recovery ($β=1.08$, $p<.01$). By integrating these approaches, we offer a nuanced understanding of suicidality, bridging digital behavior and clinical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11678v4">Which Type of Students can LLMs Act? Investigating Authentic Simulation with Graph-based Human-AI Collaborative System</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 This work has been submitted to AI Open for possible publication
    </div>
    <details class="paper-abstract">
      While rapid advances in large language models (LLMs) are reshaping data-driven intelligent education, accurately simulating students remains an important but challenging bottleneck for scalable educational data collection, evaluation, and intervention design. However, current works are limited by scarce real interaction data, costly expert evaluation for realism, and a lack of large-scale, systematic analyses of LLMs ability in simulating students. We address this gap by presenting a three-stage LLM-human collaborative pipeline to automatically generate and filter high-quality student agents. We leverage a two-round automated scoring validated by human experts and deploy a score propagation module to obtain more consistent scores across the student similarity graph. Experiments show that combining automated scoring, expert calibration, and graph-based propagation yields simulated student that more closely track authentication by human judgments. We then analyze which profiles and behaviors are simulated more faithfully, supporting subsequent studies on personalized learning and educational assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05497v2">Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Large-scale Mixture of Experts (MoE) Large Language Models (LLMs) have recently become the frontier open weight models, achieving remarkable model capability similar to proprietary ones. But their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit LLM serving systems. To understand the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across four state-of-the-art large-scale MoE models released in 2025 (200B-1000B) using over 24,000 requests spanning diverse workloads. We perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. With our insights, we then demonstrate how to improve wafer-scale GPUs as a case study, and show that minor architectural modifications leveraging the insights achieve substantial performance gains, delivering 5.3x and 3.1x average speedups on DeepSeek V3 and Qwen3, respectively. Our work presents the first comprehensive data-centric analysis of large-scale MoE models and a concrete design study using the learned lessons, with profiling traces and simulation framework already open-sourced with $>$1k downloads. Our traces and results are publicly available at https://huggingface.co/datasets/core12345/MoE_expert_selection_trace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04473v2">Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, an LLM-powered framework for generating high-quality Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over questions. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models.We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.19159v4">Sliding-Window Merging for Compacting Patch-Redundant Layers in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Depth-wise pruning accelerates LLM inference in resource-constrained scenarios but suffers from performance degradation due to direct removal of entire Transformer layers. This paper reveals ``Patch-like'' redundancy across layers via correlation analysis of the outputs of different layers in reproducing kernel Hilbert space, demonstrating consecutive layers exhibit high functional similarity. Building on this observation, this paper proposes Sliding-Window Merging (SWM) - a dynamic compression method that selects consecutive layers from top to bottom using a pre-defined similarity threshold, and compacts patch-redundant layers through a parameter consolidation, thereby simplifying the model structure while maintaining its performance. Extensive experiments on LLMs with various architectures and different parameter scales show that our method outperforms existing pruning techniques in both zero-shot inference performance and retraining recovery quality after pruning. In particular, in the experiment with 35% pruning on the Vicuna-7B model, our method achieved a 1.654% improvement in average performance on zero-shot tasks compared to the existing method. Moreover, we further reveal the potential of combining depth pruning with width pruning to enhance the pruning effect. Our codes are available at https://github.com/920927/SLM-a-sliding-layer-merging-method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.23749v2">A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping automated program repair. We present a unified taxonomy that groups 62 recent LLM-based repair systems into four paradigms defined by parameter adaptation and control authority over the repair loop, and overlays two cross-cutting layers for retrieval and analysis augmentation. Prior surveys have either focused on classical software repair techniques, on LLMs in software engineering more broadly, or on subsets of LLM-based software repair, such as fine-tuning strategies or vulnerability repair. We complement these works by treating fine-tuning, prompting, procedural pipelines, and agentic frameworks as first-class paradigms and systematically mapping representative systems to each of these paradigms. We also consolidate evaluation practice on common benchmarks by recording benchmark scope, pass@k, and fault-localization assumptions to support a more meaningful comparison of reported success rates. We clarify trade-offs among paradigms in task alignment, deployment cost, controllability, and ability to repair multi-hunk or cross-file bugs. We discuss challenges in current LLM-based software repair and outline research directions. Our artifacts, including the representation papers and scripted survey pipeline, are publicly available at https://github.com/GLEAM-Lab/ProgramRepair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08833v3">An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 34 pages, 9 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 51.5% on the originals but drops by 4.7 percentage points on surface-renaming variants, and by 12.9 percentage points on parametric variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18874v2">When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04518v1">UW-BioNLP at ChemoTimelines 2025: Thinking, Fine-Tuning, and Dictionary-Enhanced LLM Systems for Chemotherapy Timeline Extraction</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 To be published in Proceedings of the 7th Clinical Natural Language Processing Workshop
    </div>
    <details class="paper-abstract">
      The ChemoTimelines shared task benchmarks methods for constructing timelines of systemic anticancer treatment from electronic health records of cancer patients. This paper describes our methods, results, and findings for subtask 2 -- generating patient chemotherapy timelines from raw clinical notes. We evaluated strategies involving chain-of-thought thinking, supervised fine-tuning, direct preference optimization, and dictionary-based lookup to improve timeline extraction. All of our approaches followed a two-step workflow, wherein an LLM first extracted chemotherapy events from individual clinical notes, and then an algorithm normalized and aggregated events into patient-level timelines. Each specific method differed in how the associated LLM was utilized and trained. Multiple approaches yielded competitive performances on the test set leaderboard, with fine-tuned Qwen3-14B achieving the best official score of 0.678. Our results and analyses could provide useful insights for future attempts on this task as well as the design of similar tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.15555v3">Bayesian Concept Bottleneck Models with LLM Priors</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 2025 Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) have been proposed as a compromise between white-box and black-box models, aiming to achieve interpretability without sacrificing accuracy. The standard training procedure for CBMs is to predefine a candidate set of human-interpretable concepts, extract their values from the training data, and identify a sparse subset as inputs to a transparent prediction model. However, such approaches are often hampered by the tradeoff between exploring a sufficiently large set of concepts versus controlling the cost of obtaining concept extractions, resulting in a large interpretability-accuracy tradeoff. This work investigates a novel approach that sidesteps these challenges: BC-LLM iteratively searches over a potentially infinite set of concepts within a Bayesian framework, in which Large Language Models (LLMs) serve as both a concept extraction mechanism and prior. Even though LLMs can be miscalibrated and hallucinate, we prove that BC-LLM can provide rigorous statistical inference and uncertainty quantification. Across image, text, and tabular datasets, BC-LLM outperforms interpretable baselines and even black-box models in certain settings, converges more rapidly towards relevant concepts, and is more robust to out-of-distribution samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04474v1">LLM-SrcLog: Towards Proactive and Unified Log Template Extraction via Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Log parsing transforms raw logs into structured templates containing constants and variables. It underpins anomaly detection, failure diagnosis, and other AIOps tasks. Current parsers are mostly reactive and log-centric. They only infer templates from logs, mostly overlooking the source code. This restricts their capacity to grasp dynamic log structures or adjust to evolving systems. Moreover, per-log LLM inference is too costly for practical deployment. In this paper, we propose LLM-SrcLog, a proactive and unified framework for log template parsing. It extracts templates directly from source code prior to deployment and supplements them with data-driven parsing for logs without available code. LLM-SrcLog integrates a cross-function static code analyzer to reconstruct meaningful logging contexts, an LLM-based white-box template extractor with post-processing to distinguish constants from variables, and a black-box template extractor that incorporates data-driven clustering for remaining unmatched logs. Experiments on two public benchmarks (Hadoop and Zookeeper) and a large-scale industrial system (Sunfire-Compute) show that, compared to two LLM-based baselines, LLM-SrcLog improves average F1-score by 2-17% and 8-35%. Meanwhile, its online parsing latency is comparable to data-driven methods and about 1,000 times faster than per-log LLM parsing. LLM-SrcLog achieves a near-ideal balance between speed and accuracy. Finally, we further validate the effectiveness of LLM-SrcLog through practical case studies in a real-world production environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13725v2">AI Kill Switch for malicious web-based LLM agent</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Recently, web-based Large Language Model (LLM) agents autonomously perform increasingly complex tasks, thereby bringing significant convenience. However, they also amplify the risks of malicious misuse cases such as unauthorized collection of personally identifiable information (PII), generation of socially divisive content, and even automated web hacking. To address these threats, we propose an AI Kill Switch technique that can immediately halt the operation of malicious web-based LLM agents. To achieve this, we introduce AutoGuard - the key idea is generating defensive prompts that trigger the safety mechanisms of malicious LLM agents. In particular, generated defense prompts are transparently embedded into the website's DOM so that they remain invisible to human users but can be detected by the crawling process of malicious agents, triggering its internal safety mechanisms to abort malicious actions once read. To evaluate our approach, we constructed a dedicated benchmark consisting of three representative malicious scenarios. Experimental results show that AutoGuard achieves over 80% Defense Success Rate (DSR) across diverse malicious agents, including GPT-4o, Claude-4.5-Sonnet and generalizes well to advanced models like GPT-5.1, Gemini-2.5-flash, and Gemini-3-pro. Also, our approach demonstrates robust defense performance in real-world website environments without significant performance degradation for benign agents. Through this research, we demonstrate the controllability of web-based LLM agents, thereby contributing to the broader effort of AI control and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04419v1">Solving LLM Repetition Problem in Production: A Comprehensive Study of Multiple Solutions</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      The repetition problem, where Large Language Models (LLMs) continuously generate repetitive content without proper termination, poses a critical challenge in production deployments, causing severe performance degradation and system stalling. This paper presents a comprehensive investigation and multiple practical solutions for the repetition problem encountered in real-world batch code interpretation tasks. We identify three distinct repetition patterns: (1) business rule generation repetition, (2) method call relationship analysis repetition, and (3) PlantUML diagram syntax generation repetition. Through rigorous theoretical analysis based on Markov models, we establish that the root cause lies in greedy decoding's inability to escape repetitive loops, exacerbated by self-reinforcement effects. Our comprehensive experimental evaluation demonstrates three viable solutions: (1) Beam Search decoding with early_stopping=True serves as a universal post-hoc mechanism that effectively resolves all three repetition patterns; (2) presence_penalty hyperparameter provides an effective solution specifically for BadCase 1; and (3) Direct Preference Optimization (DPO) fine-tuning offers a universal model-level solution for all three BadCases. The primary value of this work lies in combining first-hand production experience with extensive experimental validation. Our main contributions include systematic theoretical analysis of repetition mechanisms, comprehensive evaluation of multiple solutions with task-specific applicability mapping, identification of early_stopping as the critical parameter for Beam Search effectiveness, and practical production-ready solutions validated in real deployment environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04416v1">GovBench: Benchmarking LLM Agents for Real-World Data Governance Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Equal contribution: Zhou Liu and Zhaoyang Han. Corresponding authors: Yuanfeng Song and Wentao Zhang
    </div>
    <details class="paper-abstract">
      Data governance ensures data quality, security, and compliance through policies and standards, a critical foundation for scaling modern AI development. Recently, large language models (LLMs) have emerged as a promising solution for automating data governance by translating user intent into executable transformation code. However, existing benchmarks for automated data science often emphasize snippet-level coding or high-level analytics, failing to capture the unique challenge of data governance: ensuring the correctness and quality of the data itself. To bridge this gap, we introduce GovBench, a benchmark featuring 150 diverse tasks grounded in real-world scenarios, built on data from actual cases. GovBench employs a novel "reversed-objective" methodology to synthesize realistic noise and utilizes rigorous metrics to assess end-to-end pipeline reliability. Our analysis on GovBench reveals that current models struggle with complex, multi-step workflows and lack robust error-correction mechanisms. Consequently, we propose DataGovAgent, a framework utilizing a Planner-Executor-Evaluator architecture that integrates constraint-based planning, retrieval-augmented generation, and sandboxed feedback-driven debugging. Experimental results show that DataGovAgent significantly boosts the Average Task Score (ATS) on complex tasks from 39.7 to 54.9 and reduces debugging iterations by over 77.9 percent compared to general-purpose baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04408v1">Executable Governance for AI: Translating Policies into Rules Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Accepted to AAAI-26 AI Governance Workshop (in-person presentation); 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      AI policy guidance is predominantly written as prose, which practitioners must first convert into executable rules before frameworks can evaluate or enforce them. This manual step is slow, error-prone, difficult to scale, and often delays the use of safeguards in real-world deployments. To address this gap, we present Policy-to-Tests (P2T), a framework that converts natural-language policy documents into normalized, machine-readable rules. The framework comprises a pipeline and a compact domain-specific language (DSL) that encodes hazards, scope, conditions, exceptions, and required evidence, yielding a canonical representation of extracted rules. To test the framework beyond a single policy, we apply it across general frameworks, sector guidance, and enterprise standards, extracting obligation-bearing clauses and converting them into executable rules. These AI-generated rules closely match strong human baselines on span-level and rule-level metrics, with robust inter-annotator agreement on the gold set. To evaluate downstream behavioral and safety impact, we add HIPAA-derived safeguards to a generative agent and compare it with an otherwise identical agent without guardrails. An LLM-based judge, aligned with gold-standard criteria, measures violation rates and robustness to obfuscated and compositional prompts. Detailed results are provided in the appendix. We release the codebase, DSL, prompts, and rule sets as open-source resources to enable reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03762v2">RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Automatic Heuristic Design (AHD) has gained traction as a promising solution for solving combinatorial optimization problems (COPs). Large Language Models (LLMs) have emerged and become a promising approach to achieving AHD, but current LLM-based AHD research often only considers a single role. This paper proposes RoCo, a novel Multi-Agent Role-Based System, to enhance the diversity and quality of AHD through multi-role collaboration. RoCo coordinates four specialized LLM-guided agents-explorer, exploiter, critic, and integrator-to collaboratively generate high-quality heuristics. The explorer promotes long-term potential through creative, diversity-driven thinking, while the exploiter focuses on short-term improvements via conservative, efficiency-oriented refinements. The critic evaluates the effectiveness of each evolution step and provides targeted feedback and reflection. The integrator synthesizes proposals from the explorer and exploiter, balancing innovation and exploitation to drive overall progress. These agents interact in a structured multi-round process involving feedback, refinement, and elite mutations guided by both short-term and accumulated long-term reflections. We evaluate RoCo on five different COPs under both white-box and black-box settings. Experimental results demonstrate that RoCo achieves superior performance, consistently generating competitive heuristics that outperform existing methods including ReEvo and HSEvo, both in white-box and black-box scenarios. This role-based collaborative paradigm establishes a new standard for robust and high-performing AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04359v1">Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04356v1">Mitigating Object and Action Hallucinations in Multimodal LLMs via Self-Augmented Contrastive Alignment</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026. Project page: https://kpc0810.github.io/santa/
    </div>
    <details class="paper-abstract">
      Recent advancement in multimodal LLMs (MLLMs) has demonstrated their remarkable capability to generate descriptive captions for input videos. However, these models suffer from factual inaccuracies in the generated descriptions, causing severe hallucination issues. While prior works have explored alleviating hallucinations for static images, jointly mitigating visual object and temporal action hallucinations for dynamic videos remains a challenging and unsolved task. To tackle this challenge, we propose a Self-Augmented Contrastive Alignment (SANTA) framework for enabling object and action faithfulness by exempting the spurious correlations and enforcing the emphasis on visual facts. SANTA employs a hallucinative self-augmentation scheme to identify the potential hallucinations that lie in the MLLM and transform the original captions to the contrasted negatives. Furthermore, we develop a tracklet-phrase contrastive alignment to match the regional objects and relation-guided actions with their corresponding visual and temporal phrases. Extensive experiments demonstrate that SANTA outperforms existing methods in alleviating object and action hallucinations, yielding superior performance on the hallucination examination benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04355v1">Counting Without Running: Evaluating LLMs' Reasoning About Code Complexity</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 13 pages, 6 figures, MLSys 2026 Submission
    </div>
    <details class="paper-abstract">
      Modern GPU software stacks demand developers who can anticipate performance bottlenecks before ever launching a kernel; misjudging floating-point workloads upstream can derail tuning, scheduling, and even hardware procurement. Yet despite rapid progress in code generation, today's Large Language Models (LLMs) are rarely tested on this kind of forward-looking reasoning. We close that gap with gpuFLOPBench, a benchmark that asks models to "count without running" by predicting single and double-precision FLOP counts for 577 CUDA kernels drawn from HeCBench, annotated with ground-truth profiles and eight execution attributes that distinguish trivially analyzable code from kernels whose FLOPs depend on hidden compiler or runtime behavior. Evaluating current closed-source reasoning models shows clear but uneven progress: the newest LLMs achieve perfect classification on straightforward kernels but still incur multiple order-of-magnitude errors whenever implicit FLOPs arise from division, intrinsic math functions, or common subexpressions. These results surface a core limitation of existing code assistants -- the inability to internalize hardware-specific microcode effects -- and position gpuFLOPBench as a focused testbed for developing LLM tooling that can reason about performance with the same rigor as experienced GPU developers. Sources are available at our repository: https://github.com/Scientific-Computing-Lab/gpuFLOPBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04350v1">ClusterFusion: Hybrid Clustering with Embedding Guidance and LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Text clustering is a fundamental task in natural language processing, yet traditional clustering algorithms with pre-trained embeddings often struggle in domain-specific contexts without costly fine-tuning. Large language models (LLMs) provide strong contextual reasoning, yet prior work mainly uses them as auxiliary modules to refine embeddings or adjust cluster boundaries. We propose ClusterFusion, a hybrid framework that instead treats the LLM as the clustering core, guided by lightweight embedding methods. The framework proceeds in three stages: embedding-guided subset partition, LLM-driven topic summarization, and LLM-based topic assignment. This design enables direct incorporation of domain knowledge and user preferences, fully leveraging the contextual adaptability of LLMs. Experiments on three public benchmarks and two new domain-specific datasets demonstrate that ClusterFusion not only achieves state-of-the-art performance on standard tasks but also delivers substantial gains in specialized domains. To support future work, we release our newly constructed dataset and results on all benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18929v2">Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 NeurIPS 2025; 27 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a critical component of large language model (LLM) post-training. However, on-policy algorithms used for post-training are not naturally robust to a diversified content of experience replay buffers, which asynchronous off-policy actors can efficiently populate in parallel to training. We propose efficiently learning on such off-policy data via Trajectory Balance with Asynchrony (TBA), an approach to asynchronous RL for LLMs that leverages the principled off-policy TB objective. On math, preference-tuning, and automated red-teaming tasks, we post-train models ranging from Pythia 410M to Qwen 2.5 7B, finding TBA offers speed and performance boosts over strong baselines like Online DPO and Dr. GRPO. Beyond TBA's performance benefits (high accuracy even as asynchrony grows) and speedups ($4\times$ or more), we show its reward- and recency-prioritizing sampling enable further gains as data generation is scaled. Our code is available at https://github.com/bbartoldson/TBA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03994v1">Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted to the AAAI 2026 Deployable AI (DAI) Workshop
    </div>
    <details class="paper-abstract">
      Aligning proprietary large language models (LLMs) with internal organizational policies has become an urgent priority as organizations increasingly deploy LLMs in sensitive domains such as legal support, finance, and medical services. Beyond generic safety filters, enterprises require reliable mechanisms to detect policy violations within their regulatory and operational frameworks, where breaches can trigger legal and reputational risks. Existing content moderation frameworks, such as guardrails, remain largely confined to the safety domain and lack the robustness to capture nuanced organizational policies. LLM-as-a-judge and fine-tuning approaches, though flexible, introduce significant latency and lack interpretability. To address these limitations, we propose a training-free and efficient method that treats policy violation detection as an out-of-distribution (OOD) detection problem. Inspired by whitening techniques, we apply a linear transformation to decorrelate the model's hidden activations and standardize them to zero mean and unit variance, yielding near-identity covariance matrix. In this transformed space, we use the Euclidean norm as a compliance score to detect policy violations. The method requires only the policy text and a small number of illustrative samples, which makes it light-weight and easily deployable. On a challenging policy benchmark, our approach achieves state-of-the-art results, surpassing both existing guardrails and fine-tuned reasoning models. This work provides organizations with a practical and statistically grounded framework for policy-aware oversight of LLMs, advancing the broader goal of deployable AI governance. Code is available at: https://tinyurl.com/policy-violation-detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.05235v2">SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain traction, their reliance on power-hungry GPUs places ever-increasing energy demands, raising environmental and monetary concerns. Inference dominates LLM workloads, presenting a critical challenge for providers: minimizing energy costs under Service-Level Objectives (SLOs) that ensure optimal user experience. In this paper, we present \textit{throttLL'eM}, a framework that reduces energy consumption while meeting SLOs through the use of instance and GPU frequency scaling. \textit{throttLL'eM} features mechanisms that project future KV cache usage and batch size. Leveraging a Machine-Learning (ML) model that receives these projections as inputs, \textit{throttLL'eM} manages performance at the iteration level to satisfy SLOs with reduced frequencies and instance sizes. We show that the proposed ML model achieves $R^2$ scores greater than 0.97 and miss-predicts performance by less than 1 iteration per second on average. Experimental results on LLM inference traces show that \textit{throttLL'eM} achieves up to 43.8\% lower energy consumption and an energy efficiency improvement of at least $1.71\times$ under SLOs, when compared to NVIDIA's Triton server.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03111v3">Les Dissonances: Cross-Tool Harvesting and Polluting in Pool-of-Tools Empowered LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03847v1">DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has shown strong performance in LLM post-training, but real-world deployment often involves noisy or incomplete supervision. In such settings, complex and unreliable supervision signals can destabilize training and harm generalization. While existing approaches such as worst-case optimization (e.g., RFQI, CQL) and mean-based methods (e.g., PPO, GRPO) can improve stability, they often overlook generalization and may produce overly conservative policies, leading to uneven performance across diverse real scenarios. To this end, we introduce DVPO (Distributional Value Modeling with Risk-aware Policy Optimization), a new RL framework that combines conditional risk theory with distributional value modeling to better balance robustness and generalization. DVPO learns token-level value distributions to provide fine-grained supervision, and applies an asymmetric risk regularization to shape the distribution tails: it contracts the lower tail to dampen noisy negative deviations, while expanding the upper tail to preserve exploratory diversity. Across extensive experiments and analysis in multi-turn dialogue, math reasoning, and scientific QA, DVPO consistently outperforms PPO, GRPO, and robust Bellman-based PPO under noisy supervision, showing its potential for LLM post-training in the real-world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03838v1">Training and Evaluation of Guideline-Based Medical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to follow medical consensus guidelines step-by-step in their reasoning and prediction process. Since consensus guidelines are ubiquitous in medicine, instantiations of verbalized medical inference rules to electronic health records provide data for fine-tuning LLMs to learn consensus rules and possible exceptions thereof for many medical areas. Consensus rules also enable an automatic evaluation of the model's inference process regarding its derivation correctness (evaluating correct and faithful deduction of a conclusion from given premises) and value correctness (comparing predicted values against real-world measurements). We exemplify our work using the complex Sepsis-3 consensus definition. Our experiments show that small fine-tuned models outperform one-shot learning of considerably larger LLMs that are prompted with the explicit definition and models that are trained on medical texts including consensus definitions. Since fine-tuning on verbalized rule instantiations of a specific medical area yields nearly perfect derivation correctness for rules (and exceptions) on unseen patient data in that area, the bottleneck for early prediction is not out-of-distribution generalization, but the orthogonal problem of generalization into the future by forecasting sparsely and irregularly sampled clinical variables. We show that the latter results can be improved by integrating the output representations of a time series forecasting model with the LLM in a multimodal setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04964v5">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02901v2">Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in $\{\pm 1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence, yet their massive memory and computational demands necessitate aggressive quantization, increasingly pushing representations toward the theoretical limit of a single bit. While complex-valued LLMs, such as iFairy, offer a superior chance for low-bit representation compared to real-valued counterparts, they require training from scratch, preventing the utilization of the vast ecosystem of pre-trained real-valued foundation models. Here we present Fairy2i, a universal framework that transforms pre-trained real-valued layers into an equivalent widely-linear complex form, enabling extremely low-bit quantization while reusing existing checkpoints. By proving a lossless mathematical equivalence between real and widely-linear maps, we convert standard Transformers into the complex domain and employ a phase-aware quantization scheme with a highly efficient codebook of fourth roots of unity. Furthermore, we introduce a recursive residual quantization mechanism that iteratively minimizes quantization error, allowing inference to proceed via efficient multiplication-free accumulation. We demonstrate that Fairy2i restores the performance of LLaMA-2 7B at an effective 2-bit precision to levels nearly comparable with full-precision baselines, significantly outperforming state-of-the-art real-valued binary and ternary quantization methods. This work bridges the gap between the representational efficiency of complex-valued arithmetic and the practical utility of pre-trained models, paving a new way for efficient inference on commodity hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03816v1">Log Probability Tracking of LLM APIs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      When using an LLM through an API provider, users expect the served model to remain consistent over time, a property crucial for the reliability of downstream applications and the reproducibility of research. Existing audit methods are too costly to apply at regular time intervals to the wide range of available LLM APIs. This means that model updates are left largely unmonitored in practice. In this work, we show that while LLM log probabilities (logprobs) are usually non-deterministic, they can still be used as the basis for cost-effective continuous monitoring of LLM APIs. We apply a simple statistical test based on the average value of each token logprob, requesting only a single token of output. This is enough to detect changes as small as one step of fine-tuning, making this approach more sensitive than existing methods while being 1,000x cheaper. We introduce the TinyChange benchmark as a way to measure the sensitivity of audit methods in the context of small, realistic model changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03762v1">RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Automatic Heuristic Design (AHD) has gained traction as a promising solution for solving combinatorial optimization problems (COPs). Large Language Models (LLMs) have emerged and become a promising approach to achieving AHD, but current LLM-based AHD research often only considers a single role. This paper proposes RoCo, a novel Multi-Agent Role-Based System, to enhance the diversity and quality of AHD through multi-role collaboration. RoCo coordinates four specialized LLM-guided agents-explorer, exploiter, critic, and integrator-to collaboratively generate high-quality heuristics. The explorer promotes long-term potential through creative, diversity-driven thinking, while the exploiter focuses on short-term improvements via conservative, efficiency-oriented refinements. The critic evaluates the effectiveness of each evolution step and provides targeted feedback and reflection. The integrator synthesizes proposals from the explorer and exploiter, balancing innovation and exploitation to drive overall progress. These agents interact in a structured multi-round process involving feedback, refinement, and elite mutations guided by both short-term and accumulated long-term reflections. We evaluate RoCo on five different COPs under both white-box and black-box settings. Experimental results demonstrate that RoCo achieves superior performance, consistently generating competitive heuristics that outperform existing methods including ReEvo and HSEvo, both in white-box and black-box scenarios. This role-based collaborative paradigm establishes a new standard for robust and high-performing AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03737v1">AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering powerful semantic understanding to bridge this gap. Despite their potential, deploying LLMs in this high-stakes domain is fraught with challenges, including factual hallucinations, specialized knowledge gaps, and high operational costs. To overcome these barriers, we introduce \textbf{AR-Med}, a novel framework for \textbf{A}utomated \textbf{R}elevance assessment for \textbf{Med}ical search that has been successfully deployed at scale on the Online Medical Delivery Platforms. AR-Med grounds LLM reasoning in verified medical knowledge through a retrieval-augmented approach, ensuring high accuracy and reliability. To enable efficient online service, we design a practical knowledge distillation scheme that compresses large teacher models into compact yet powerful student models. We also introduce LocalQSMed, a multi-expert annotated benchmark developed to guide model iteration and ensure strong alignment between offline and online performance. Extensive experiments show AR-Med achieves an offline accuracy of over 93\%, a 24\% absolute improvement over the original online system, and delivers significant gains in online relevance and user satisfaction. Our work presents a practical and scalable blueprint for developing trustworthy, LLM-powered systems in real-world healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03720v1">Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at https://github.com/S2AILab/CAHL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17701v2">Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Published in CEUR Workshop Proceedings, Vol. 4114, edu4AI'25: 2nd Workshop on Education for Artificial Intelligence, co-located with ECAI 2025, Bologna, Italy
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for educational support, yet their response quality varies depending on the language of interaction. This paper presents an automated multilingual pipeline for generating, solving, and evaluating math problems aligned with the German K-10 curriculum. We generated 628 math exercises and translated them into English, German, and Arabic. Three commercial LLMs (GPT-4o-mini, Gemini 2.5 Flash, and Qwen-plus) were prompted to produce step-by-step solutions in each language. A held-out panel of LLM judges, including Claude 3.5 Haiku, evaluated solution quality using a comparative framework. Results show a consistent gap, with English solutions consistently rated highest, and Arabic often ranked lower. These findings highlight persistent linguistic bias and the need for more equitable multilingual AI systems in education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.09245v2">Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted to AAAI 2026 (Main Technical Track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in automating data science workflows, but existing models still struggle with multi-step reasoning and tool use, which limits their effectiveness on complex data analysis tasks. To address this, we propose a scalable pipeline that extracts high-quality, tool-based data analysis tasks and their executable multi-step solutions from real-world Jupyter notebooks and associated data files. Using this pipeline, we introduce NbQA, a large-scale dataset of standardized task-solution pairs that reflect authentic tool-use patterns in practical data science scenarios. To further enhance multi-step reasoning, we present Jupiter, a framework that formulates data analysis as a search problem and applies Monte Carlo Tree Search (MCTS) to generate diverse solution trajectories for value model learning. During inference, Jupiter combines the value model and node visit counts to efficiently collect executable multi-step plans with minimal search steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench, respectively-matching or surpassing GPT-4o and advanced agent frameworks. Further evaluations demonstrate improved generalization and stronger tool-use reasoning across diverse multi-step reasoning tasks. Code and data are available at https://github.com/microsoft/Jupiter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03620v1">SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at https://github.com/HanxiuZhang/SELF_v2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00926v3">LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 19 pages, 6 figures, 28 models tested across 4,200 trials
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) grow in capability, do they develop self-awareness as an emergent behavior? And if so, can we measure it? We introduce the AI Self-Awareness Index (AISAI), a game-theoretic framework for measuring self-awareness through strategic differentiation. Using the "Guess 2/3 of Average" game, we test 28 models (OpenAI, Anthropic, Google) across 4,200 trials with three opponent framings: (A) against humans, (B) against other AI models, and (C) against AI models like you. We operationalize self-awareness as the capacity to differentiate strategic reasoning based on opponent type. Finding 1: Self-awareness emerges with model advancement. The majority of advanced models (21/28, 75%) demonstrate clear self-awareness, while older/smaller models show no differentiation. Finding 2: Self-aware models rank themselves as most rational. Among the 21 models with self-awareness, a consistent rationality hierarchy emerges: Self > Other AIs > Humans, with large AI attribution effects and moderate self-preferencing. These findings reveal that self-awareness is an emergent capability of advanced LLMs, and that self-aware models systematically perceive themselves as more rational than humans. This has implications for AI alignment, human-AI collaboration, and understanding AI beliefs about human capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18098v2">Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Published at NeurIPS 2025; 18 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01513v2">SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08863v3">GeoJSON Agents:A Multi-Agent LLM Architecture for Geospatial Analysis-Function Calling vs Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated substantial progress in task automation and natural language understanding. However, without domain expertise in geographic information science (GIS), they continue to encounter limitations including reduced accuracy and unstable performance when processing complex tasks. To address these challenges, we propose GeoJSON Agents-a novel multi-agent LLM architecture specifically designed for geospatial analysis. This framework transforms natural language instructions into structured GeoJSON operations through two LLM enhancement techniques: Function Calling and Code Generation. The architecture integrates three core components: task parsing, agent collaboration, and result integration. The Planner agent systematically decomposes user-defined tasks into executable subtasks, while Worker agents perform spatial data processing and analysis either by invoking predefined function APIs or by generating and executing Python-based analytical code. The system produces reusable, standards-compliant GeoJSON outputs through iterative refinement. To evaluate both approaches, we constructed a benchmark comprising 70 tasks spanning basic, intermediate, and advanced complexity levels, conducting experiments with OpenAI's GPT-4o as the core model. Results indicate that the Code Generation-based agent achieved 97.14% accuracy, while the Function Calling-based agent attained 85.71%-both significantly outperforming the best-performing general-purpose model (48.57%). Comparative analysis reveals Code Generation offers superior flexibility for complex, open-ended tasks, whereas Function Calling provides enhanced execution stability for structured operations. This study represents the first systematic integration of GeoJSON data with a multi-agent LLM framework and provides empirical evidence comparing two mainstream enhancement methodologies in geospatial context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03503v1">Understanding LLM Reasoning for Abstractive Summarization</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 26 pages,15 figures
    </div>
    <details class="paper-abstract">
      While the reasoning capabilities of Large Language Models (LLMs) excel in analytical tasks such as mathematics and code generation, their utility for abstractive summarization remains widely assumed but largely unverified. To bridge this gap, we first tailor general reasoning strategies to the summarization domain. We then conduct a systematic, large scale comparative study of 8 reasoning strategies and 3 Large Reasoning Models (LRMs) across 8 diverse datasets, assessing both summary quality and faithfulness. Our findings show that reasoning is not a universal solution and its effectiveness is highly dependent on the specific strategy and context. Specifically, we observe a trade-off between summary quality and factual faithfulness: explicit reasoning strategies tend to improve fluency at the expense of factual grounding, while implicit reasoning in LRMs exhibits the inverse pattern. Furthermore, increasing an LRM's internal reasoning budget does not improve, and can even hurt, factual consistency, suggesting that effective summarization demands faithful compression rather than creative over-thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03501v1">SocraticAI: Transforming LLMs into Guided CS Tutors Through Scaffolded Interaction</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Presented in the Best Practices Track of COMPUTE 2025 (arXiv:2512.02349)
    </div>
    <details class="paper-abstract">
      We present SocraticAI, a scaffolded AI tutoring system that integrates large language models (LLMs) into undergraduate Computer Science education through structured constraints rather than prohibition. The system enforces well-formulated questions, reflective engagement, and daily usage limits while providing Socratic dialogue scaffolds. Unlike traditional AI bans, our approach cultivates responsible and strategic AI interaction skills through technical guardrails, including authentication, query validation, structured feedback, and RAG-based course grounding. Initial deployment demonstrates that students progress from vague help-seeking to sophisticated problem decomposition within 2-3 weeks, with over 75% producing substantive reflections and displaying emergent patterns of deliberate, strategic AI use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04847v2">Grounded Test-Time Adaptation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions. This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time. To address these issues, we propose two distinct and complementary strategies for adapting LLM agents by leveraging environment-specific information available during deployment. First, an online distributional adaptation method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format. Second, a deployment-time dynamics grounding method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with a nonparametric world model. We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation. Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost. We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents. For example, on the WebArena multi-site split, this method increases the agent's success rate from 2% to 23%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03444v1">PerFACT: Motion Policy with LLM-Powered Dataset Synthesis and Fusion Action-Chunking Transformers</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Deep learning methods have significantly enhanced motion planning for robotic manipulators by leveraging prior experiences within planning datasets. However, state-of-the-art neural motion planners are primarily trained on small datasets collected in manually generated workspaces, limiting their generalizability to out-of-distribution scenarios. Additionally, these planners often rely on monolithic network architectures that struggle to encode critical planning information. To address these challenges, we introduce Motion Policy with Dataset Synthesis powered by large language models (LLMs) and Fusion Action-Chunking Transformers (PerFACT), which incorporates two key components. Firstly, a novel LLM-powered workspace generation method, MotionGeneralizer, enables large-scale planning data collection by producing a diverse set of semantically feasible workspaces. Secondly, we introduce Fusion Motion Policy Networks (MpiNetsFusion), a generalist neural motion planner that uses a fusion action-chunking transformer to better encode planning signals and attend to multiple feature modalities. Leveraging MotionGeneralizer, we collect 3.5M trajectories to train and evaluate MpiNetsFusion against state-of-the-art planners, which shows that the proposed MpiNetsFusion can plan several times faster on the evaluated tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03439v1">LLM as Explainable Re-Ranker for Recommendation System</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in recommendation systems has recently gained traction. Traditional recommendation systems often lack explainability and suffer from issues such as popularity bias. Previous research has also indicated that LLMs, when used as standalone predictors, fail to achieve accuracy comparable to traditional models. To address these challenges, we propose to use LLM as an explainable re-ranker, a hybrid approach that combines traditional recommendation models with LLMs to enhance both accuracy and interpretability. We constructed a dataset to train the re-ranker LLM and evaluated the alignment between the generated dataset and human expectations. Leveraging a two-stage training process, our model significantly improved NDCG, a key ranking metric. Moreover, the re-ranker outperformed a zero-shot baseline in ranking accuracy and interpretability. These results highlight the potential of integrating traditional recommendation models with LLMs to address limitations in existing systems and pave the way for more explainable and fair recommendation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03420v1">HarnessAgent: Scaling Automatic Fuzzing Harness Construction with Tool-Augmented LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based techniques have achieved notable progress in generating harnesses for program fuzzing. However, applying them to arbitrary functions (especially internal functions) \textit{at scale} remains challenging due to the requirement of sophisticated contextual information, such as specification, dependencies, and usage examples. State-of-the-art methods heavily rely on static or incomplete context provisioning, causing failure of generating functional harnesses. Furthermore, LLMs tend to exploit harness validation metrics, producing plausible yet logically useless code. % Therefore, harness generation across large and diverse projects continues to face challenges in reliable compilation, robust code retrieval, and comprehensive validation. To address these challenges, we present HarnessAgent, a tool-augmented agentic framework that achieves fully automated, scalable harness construction over hundreds of OSS-Fuzz targets. HarnessAgent introduces three key innovations: 1) a rule-based strategy to identify and minimize various compilation errors; 2) a hybrid tool pool for precise and robust symbol source code retrieval; and 3) an enhanced harness validation pipeline that detects fake definitions. We evaluate HarnessAgent on 243 target functions from OSS-Fuzz projects (65 C projects and 178 C++ projects). It improves the three-shot success rate by approximately 20\% compared to state-of-the-art techniques, reaching 87\% for C and 81\% for C++. Our one-hour fuzzing results show that more than 75\% of the harnesses generated by HarnessAgent increase the target function coverage, surpassing the baselines by over 10\%. In addition, the hybrid tool-pool system of HarnessAgent achieves a response rate of over 90\% for source code retrieval, outperforming Fuzz Introspector by more than 30\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03418v1">YOLOA: Real-Time Affordance Detection via LLM Adapter</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 13 pages, 9 figures, conference
    </div>
    <details class="paper-abstract">
      Affordance detection aims to jointly address the fundamental "what-where-how" challenge in embodied AI by understanding "what" an object is, "where" the object is located, and "how" it can be used. However, most affordance learning methods focus solely on "how" objects can be used while neglecting the "what" and "where" aspects. Other affordance detection methods treat object detection and affordance learning as two independent tasks, lacking effective interaction and real-time capability. To overcome these limitations, we introduce YOLO Affordance (YOLOA), a real-time affordance detection model that jointly handles these two tasks via a large language model (LLM) adapter. Specifically, YOLOA employs a lightweight detector consisting of object detection and affordance learning branches refined through the LLM Adapter. During training, the LLM Adapter interacts with object and affordance preliminary predictions to refine both branches by generating more accurate class priors, box offsets, and affordance gates. Experiments on our relabeled ADG-Det and IIT-Heat benchmarks demonstrate that YOLOA achieves state-of-the-art accuracy (52.8 / 73.1 mAP on ADG-Det / IIT-Heat) while maintaining real-time performance (up to 89.77 FPS, and up to 846.24 FPS for the lightweight variant). This indicates that YOLOA achieves an excellent trade-off between accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20412v2">Structuring Collective Action with LLM-Guided Evolution: From Ill-Structured Problems to Executable Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Collective action problems, which require aligning individual incentives with collective goals, are classic examples of Ill-Structured Problems (ISPs). For an individual agent, the causal links between local actions and global outcomes are unclear, stakeholder objectives often conflict, and no single, clear algorithm can bridge micro-level choices with macro-level welfare. We present ECHO-MIMIC, a general computational framework that converts this global complexity into a tractable, Well-Structured Problem (WSP) for each agent by discovering executable heuristics and persuasive rationales. The framework operates in two stages: ECHO (Evolutionary Crafting of Heuristics from Outcomes) evolves snippets of Python code that encode candidate behavioral policies, while MIMIC (Mechanism Inference \& Messaging for Individual-to-Collective Alignment) evolves companion natural language messages that motivate agents to adopt those policies. Both phases employ a large-language-model-driven evolutionary search: the LLM proposes diverse and context-aware code or text variants, while population-level selection retains those that maximize collective performance in a simulated environment. We demonstrate this framework on two distinct ISPs: a canonical agricultural landscape management problem and a carbon-aware EV charging time slot usage problem. Results show that ECHO-MIMIC discovers high-performing heuristics compared to baselines and crafts tailored messages that successfully align simulated agent behavior with system-level goals. By coupling algorithmic rule discovery with tailored communication, ECHO-MIMIC transforms the cognitive burden of collective action into a implementable set of agent-level instructions, making previously ill-structured problems solvable in practice and opening a new path toward scalable, adaptive policy design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03416v1">TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      The architectural shift to prefill/decode (PD) disaggregation in LLM serving improves resource utilization but struggles with the bursty nature of modern workloads. Existing autoscaling policies, often retrofitted from monolithic systems like those in AIBrix and DistServe, rely on lagging indicators such as GPU utilization or coarse-grained request counts. This results in slow reactions to load spikes, leading to significant Time-to First-Token (TTFT) and Time-Per-Output-Token (TPOT) SLO violations and costly over-provisioning. We introduce TokenScale, an autoscaling framework that resolves this performance mismatch through two innovations. First, we propose Token Velocity, a novel metric that unifies the prefill, network, and decode stages by quantifying their rate of work. As a leading indicator of system backpressure, it enables proactive scaling. Second, Convertible Decoders allow decoder GPUs to dynamically execute prefill tasks during traffic spikes, creating a rapid-response buffer that absorbs bursts and eliminates the initialization latency of new prefillers. Our evaluation on a GPU cluster with production traces shows TokenScale improves SLO attainment from 50-88% to 80-96% and reduces costs by 4-14% over state-of-the-art systems, including DistServe, BlitzScale, and AIBrix. By uniting a predictive metric with a flexible system design, TokenScale significantly boosts the performance and efficiency of disaggregated LLM serving infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23143v4">MathBode: Measuring the Stability of LLM Reasoning using Frequency Response</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      This paper presents MathBode, a dynamic diagnostic for mathematical reasoning in large language models (LLMs). Instead of one-shot accuracy, MathBode treats each parametric problem as a system: we drive a single parameter sinusoidally and fit first-harmonic responses of model outputs and exact solutions. This yields interpretable, frequency-resolved metrics -- gain (amplitude tracking) and phase (lag) -- that form Bode-style fingerprints. Across five closed-form families (linear solve, ratio/saturation, compound interest, 2x2 linear systems, similar triangles), the diagnostic surfaces systematic low-pass behavior and growing phase lag that accuracy alone obscures. We compare several models against a symbolic baseline that calibrates the instrument ($G \approx 1$, $φ\approx 0$). Results separate frontier from mid-tier models on dynamics, providing a compact, reproducible protocol that complements standard benchmarks with actionable measurements of reasoning fidelity and consistency. We open-source the dataset and code to enable further research and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09442v2">Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03389v1">Continuous Prompts: LLM-Augmented Pipeline Processing over Unstructured Streams</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Monitoring unstructured streams increasingly requires persistent, semantics-aware computation, yet today's LLM frameworks remain stateless and one-shot, limiting their usefulness for long-running analytics. We introduce Continuous Prompts (CPs), the first framework that brings LLM reasoning into continuous stream processing. CPs extend RAG to streaming settings, define continuous semantic operators, and provide multiple implementations, primarily focusing on LLM-based approaches but also reporting one embedding-based variants. Furthermore, we study two LLM-centric optimizations, tuple batching and operator fusion, to significantly improve efficiency while managing accuracy loss. Because these optimizations inherently trade accuracy for speed, we present a dynamic optimization framework that uses lightweight shadow executions and cost-aware multi-objective Bayesian optimization (MOBO) to learn throughput-accuracy frontiers and adapt plans under probing budgets. We implement CPs in the VectraFlow stream processing system. Using operator-level microbenchmarks and streaming pipelines on real datasets, we show that VectraFlow can adapt to workload dynamics, navigate accuracy-efficiency trade-offs, and sustain persistent semantic queries over evolving unstructured streams.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03383v1">UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Deploying large language model (LLM) models on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by the current device workload, adding to the uncertainty of model deployment. We introduce UniQL, a unified post-training quantization and low-rank compression framework with on-device configurable pruning rates for edge LLMs. UniQL is a general framework that integrates quantization and low-rank compression for Transformers, State Space Models (SSMs), and hybrid models to support diverse edge applications. In our proposed joint framework, we introduce an efficient structured weight-sorting method that speeds up computation by 20x, quantization-aware singular value decomposition (SVD) to minimize quantization errors, state-aware weight sorting for SSMs, and a fused rotary positional embedding (RoPE) kernel for pruned models. Our framework performs weight-sorting, fine-tuning, and quantization in the cloud in a single-pass workflow, while enabling on-device configurable pruning rates up to 35%. Our experiments show that quantized and pruned models achieve a memory reduction of 4x-5.7x and a token-throughput improvement of 2.7x-3.4x, maintaining accuracy within 5% of the original models at 15% pruning across Transformers (Llama3 and Qwen2.5), SSMs (Mamba2), and hybrid models (Nemotron-H and Bamba-v2). The code and quantized models are available at: https://github.com/enyac-group/UniQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03373v1">LLM-Generated Ads: From Personalization Parity to Persuasion Superiority</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable of generating persuasive content, understanding their effectiveness across different advertising strategies becomes critical. This paper presents a two-part investigation examining LLM-generated advertising through complementary lenses: (1) personality-based and (2) psychological persuasion principles. In our first study (n=400), we tested whether LLMs could generate personalized advertisements tailored to specific personality traits (openness and neuroticism) and how their performance compared to human experts. Results showed that LLM-generated ads achieved statistical parity with human-written ads (51.1% vs. 48.9%, p > 0.05), with no significant performance differences for matched personalities. Building on these insights, our second study (n=800) shifted focus from individual personalization to universal persuasion, testing LLM performance across four foundational psychological principles: authority, consensus, cognition, and scarcity. AI-generated ads significantly outperformed human-created content, achieving a 59.1% preference rate (vs. 40.9%, p < 0.001), with the strongest performance in authority (63.0%) and consensus (62.5%) appeals. Qualitative analysis revealed AI's advantage stems from crafting more sophisticated, aspirational messages and achieving superior visual-narrative coherence. Critically, this quality advantage proved robust: even after applying a 21.2 percentage point detection penalty when participants correctly identified AI-origin, AI ads still outperformed human ads, and 29.4% of participants chose AI content despite knowing its origin. These findings demonstrate LLMs' evolution from parity in personalization to superiority in persuasive storytelling, with significant implications for advertising practice given LLMs' near-zero marginal cost and time requirements compared to human experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11678v3">Which Type of Students can LLMs Act? Investigating Authentic Simulation with Graph-based Human-AI Collaborative System</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 This work has been submitted to AI Open for possible publication
    </div>
    <details class="paper-abstract">
      While rapid advances in large language models (LLMs) are reshaping data-driven intelligent education, accurately simulating students remains an important but challenging bottleneck for scalable educational data collection, evaluation, and intervention design. However, current works are limited by scarce real interaction data, costly expert evaluation for realism, and a lack of large-scale, systematic analyses of LLMs ability in simulating students. We address this gap by presenting a three-stage LLM-human collaborative pipeline to automatically generate and filter high-quality student agents. We leverage a two-round automated scoring validated by human experts and deploy a score propagation module to obtain more consistent scores across the student similarity graph. Experiments show that combining automated scoring, expert calibration, and graph-based propagation yields simulated student that more closely track authentication by human judgments. We then analyze which profiles and behaviors are simulated more faithfully, supporting subsequent studies on personalized learning and educational assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03360v1">From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      Logical reasoning is a core challenge in natural language understanding and a fundamental capability of artificial intelligence, underpinning scientific discovery, mathematical theorem proving, and complex decision-making. Despite the remarkable progress of large language models (LLMs), most current approaches still rely on forward reasoning paradigms, generating step-by-step rationales from premises to conclusions. However, such methods often suffer from redundant inference paths, hallucinated steps, and semantic drift, resulting in inefficient and unreliable reasoning. In this paper, we propose a novel framework, Hypothesis-driven Backward Logical Reasoning (HBLR). The core idea is to integrate confidence-aware symbolic translation with hypothesis-driven backward reasoning. In the translation phase, only high-confidence spans are converted into logical form, such as First-Order Logic (FOL), while uncertain content remains in natural language. A translation reflection module further ensures semantic fidelity by evaluating symbolic outputs and reverting lossy ones back to text when necessary. In the reasoning phase, HBLR simulates human deductive thinking by assuming the conclusion is true and recursively verifying its premises. A reasoning reflection module further identifies and corrects flawed inference steps, enhancing logical coherence. Extensive experiments on five reasoning benchmarks demonstrate that HBLR consistently outperforms strong baselines in both accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16037v2">Causal LLM Routing: End-to-End Regret Minimization from Observational Data</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      LLM routing aims to select the most appropriate model for each query, balancing competing performance metrics such as accuracy and cost across a pool of language models. Prior approaches typically adopt a decoupled strategy, where the metrics are first predicted and the model is then selected based on these estimates. This setup is prone to compounding errors and often relies on full-feedback data, where each query is evaluated by all candidate models, which is costly to obtain and maintain in practice. In contrast, we learn from observational data, which records only the outcome of the model actually deployed. We propose a causal end-to-end framework that learns routing policies by minimizing decision-making regret from observational data. To enable efficient optimization, we introduce two theoretically grounded surrogate objectives: a classification-based upper bound, and a softmax-weighted regret approximation shown to recover the optimal policy at convergence. We further extend our framework to handle heterogeneous cost preferences via an interval-conditioned architecture. Experiments on public benchmarks show that our method outperforms existing baselines, achieving state-of-the-art performance across different embedding models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03324v1">Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Memory and computation remain core bottlenecks in long-horizon LLM inference due to the quadratic cost of self-attention and the ever-growing key-value (KV) cache. Existing strategies for memory-bounded inference, such as quantization, offloading, or heuristic KV eviction, either incur high orchestration costs or rely on unreliable attention-based proxies of importance. We propose TRIM-KV, a novel approach that learns each token's intrinsic importance at creation time via a lightweight retention gate. Each gate predicts a scalar retention score that decays over time, reflecting the long-term utility of the token for a specific layer and head. Tokens with low scores are evicted when the memory budget is exceeded, ensuring that the cache always contains the most critical tokens. TRIM-KV is trained efficiently through distillation from a frozen LLM combined with a capacity loss, requiring only gate fine-tuning and adding negligible inference overhead. Across mathematical reasoning (GSM8K, MATH-500, AIME24), procedural generation (LongProc), conversational long-memory benchmarks (LongMemEval), and long-context understanding (LongBench and SCBench), TRIM-KV consistently outperforms strong eviction and learnable retrieval baselines, especially in low-memory regimes. Remarkably, it even surpasses full-cache models in some settings, showing that selective retention can serve as a form of regularization, suppressing noise from uninformative tokens. Qualitative analyses further reveal that learned retention scores align with human intuition, naturally recovering heuristics such as sink tokens, sliding windows, and gist compression without explicit design. Beyond efficiency, retention scores provide insights into layer- and head-specific roles, suggesting a new path toward LLM interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03318v1">Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Published at NeurIPS Datasets and Benchmarks 2025, 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have demonstrated impressive capabilities for social interaction and are increasingly being deployed in situations where they might engage with both human and artificial agents. These interactions represent a critical frontier for LLM-based agents, yet existing evaluation methods fail to measure how well these capabilities generalize to novel social situations. In this paper, we introduce a method for evaluating the ability of LLM-based agents to cooperate in zero-shot, mixed-motive environments using Concordia, a natural language multi-agent simulation environment. Our method measures general cooperative intelligence by testing an agent's ability to identify and exploit opportunities for mutual gain across diverse partners and contexts. We present empirical results from the NeurIPS 2024 Concordia Contest, where agents were evaluated on their ability to achieve mutual gains across a suite of diverse scenarios ranging from negotiation to collective action problems. Our findings reveal significant gaps between current agent capabilities and the robust generalization required for reliable cooperation, particularly in scenarios demanding persuasion and norm enforcement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.00409v2">Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04307v1">Evaluating Long-Context Reasoning in LLM-Based WebAgents</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted NeurIPS 25 LAW Workshop
    </div>
    <details class="paper-abstract">
      As large language model (LLM)-based agents become increasingly integrated into daily digital interactions, their ability to reason across long interaction histories becomes crucial for providing personalized and contextually aware assistance. However, the performance of these agents in long context scenarios, particularly for action-taking WebAgents operating in realistic web environments, remains largely unexplored. This paper introduces a benchmark for evaluating long context reasoning capabilities of WebAgents through sequentially dependent subtasks that require retrieval and application of information from extended interaction histories. We develop a novel evaluation framework that simulates multi-session user interactions by injecting irrelevant task trajectories between dependent subtasks, creating contexts ranging from 25,000 to 150,000 tokens. Through extensive evaluation of four popular models, Claude-3.7, GPT-4.1, Llama 4, and o4-mini, we observe a dramatic performance degradation as context length increases, with success rates dropping from 40-50\% in baseline conditions to less than 10\% in long context scenarios. Our detailed error analysis reveals that agents primarily fail due to getting stuck in loops and losing track of original task objectives. We further propose an implicit RAG approach that provides modest improvements by generating task-relevant summaries, though fundamental limitations in long context reasoning persist. These findings highlight critical challenges for deploying WebAgents in realistic, long-term user interaction scenarios and provide insights for developing more robust agent architectures capable of maintaining coherent task execution across extended contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10961v2">Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Detecting vulnerabilities in source code remains a critical yet challenging task, especially when benign and vulnerable functions share significant similarities. In this work, we introduce VulTrial, a courtroom-inspired multi-agent framework designed to identify vulnerable code and to provide explanations. It employs four role-specific agents, which are security researcher, code author, moderator, and review board. Using GPT-4o as the base LLM, VulTrial almost doubles the efficacy of prior best-performing baselines. Additionally, we show that role-specific instruction tuning with small quantities of data significantly further boosts VulTrial's efficacy. Our extensive experiments demonstrate the efficacy of VulTrial across different LLMs, including an open-source, in-house-deployable model (LLaMA-3.1-8B), as well as the high quality of its generated explanations and its ability to uncover multiple confirmed zero-day vulnerabilities in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04262v1">Catching UX Flaws in Code: Leveraging LLMs to Identify Usability Flaws at the Development Stage</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 7 pages. Published in Proceedings of the 2025 IEEE Symposium on Visual Languages and Human-Centric Computing (VL/HCC). DOI: 10.1109/VL-HCC65237.2025.00024
    </div>
    <details class="paper-abstract">
      Usability evaluations are essential for ensuring that modern interfaces meet user needs, yet traditional heuristic evaluations by human experts can be time-consuming and subjective, especially early in development. This paper investigates whether large language models (LLMs) can provide reliable and consistent heuristic assessments at the development stage. By applying Jakob Nielsen's ten usability heuristics to thirty open-source websites, we generated over 850 heuristic evaluations in three independent evaluations per site using a pipeline of OpenAI's GPT-4o. For issue detection, the model demonstrated moderate consistency, with an average pairwise Cohen's Kappa of 0.50 and an exact agreement of 84%. Severity judgments showed more variability: weighted Cohen's Kappa averaged 0.63, but exact agreement was just 56%, and Krippendorff's Alpha was near zero. These results suggest that while GPT-4o can produce internally consistent evaluations, especially for identifying the presence of usability issues, its ability to judge severity varies and requires human oversight in practice. Our findings highlight the feasibility and limitations of using LLMs for early-stage, automated usability testing, and offer a foundation for improving consistency in automated User Experience (UX) evaluation. To the best of our knowledge, our work provides one of the first quantitative inter-rater reliability analyses of automated heuristic evaluation and highlights methods for improving model consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13400v5">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04129v1">Tipping the Dominos: Topology-Aware Multi-Hop Attacks on LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MASs) have reshaped the digital landscape with their emergent coordination and problem-solving capabilities. However, current security evaluations of MASs are still confined to limited attack scenarios, leaving their security issues unclear and likely underestimated. To fill this gap, we propose TOMA, a topology-aware multi-hop attack scheme targeting MASs. By optimizing the propagation of contamination within the MAS topology and controlling the multi-hop diffusion of adversarial payloads originating from the environment, TOMA unveils new and effective attack vectors without requiring privileged access or direct agent manipulation. Experiments demonstrate attack success rates ranging from 40% to 78% across three state-of-the-art MAS architectures: \textsc{Magentic-One}, \textsc{LangManus}, and \textsc{OWL}, and five representative topologies, revealing intrinsic MAS vulnerabilities that may be overlooked by existing research. Inspired by these findings, we propose a conceptual defense framework based on topology trust, and prototype experiments show its effectiveness in blocking 94.8% of adaptive and composite attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03024v1">TokenPowerBench: Benchmarking the Power Consumption of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Accepted by the AAAI'26 Conference Main Track
    </div>
    <details class="paper-abstract">
      Large language model (LLM) services now answer billions of queries per day, and industry reports show that inference, not training, accounts for more than 90% of total power consumption. However, existing benchmarks focus on either training/fine-tuning or performance of inference and provide little support for power consumption measurement and analysis of inference. We introduce TokenPowerBench, the first lightweight and extensible benchmark designed for LLM-inference power consumption studies. The benchmark combines (i) a declarative configuration interface covering model choice, prompt set, and inference engine, (ii) a measurement layer that captures GPU-, node-, and system-level power without specialized power meters, and (iii) a phase-aligned metrics pipeline that attributes energy to the prefill and decode stages of every request. These elements make it straight-forward to explore the power consumed by an LLM inference run; furthermore, by varying batch size, context length, parallelism strategy and quantization, users can quickly assess how each setting affects joules per token and other energy-efficiency metrics. We evaluate TokenPowerBench on four of the most widely used model series (Llama, Falcon, Qwen, and Mistral). Our experiments cover from 1 billion parameters up to the frontier-scale Llama3-405B model. Furthermore, we release TokenPowerBench as open source to help users to measure power consumption, forecast operating expenses, and meet sustainability targets when deploying LLM services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03019v1">Distribution-Calibrated Inference time compute for Thinking LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Thinking Large Language Models (LLMs) used as judges for pairwise preferences remain noisy at the single-sample level, and common aggregation rules (majority vote, soft self-consistency, or instruction-based self-aggregation) are inconsistent when ties are allowed. We study inference-time compute (ITC) for evaluators that generate n independent thinking-rating samples per item, and propose a principled, distribution-calibrated aggregation scheme. Our method models three-way preferences with a Bradley-Terry-Davidson formulation on rating counts, leveraging both polarity (margin among non-ties) and decisiveness (non-tie rate) to distinguish narrow margins from strong consensus. Across various evaluation benchmarks, our approach consistently reduces MAE and increases pairwise accuracy versus standard baselines, and when evaluated against human-consensus meta-labels, matches or exceeds individual human raters. These results show that carefully allocating ITC and aggregating with distribution-aware methods turns noisy individual model judgments into reliable ratings for evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03005v1">From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01472v3">LLM-NAS: LLM-driven Hardware-Aware Neural Architecture Search</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Hardware-Aware Neural Architecture Search (HW-NAS) requires joint optimization of accuracy and latency under device constraints. Traditional supernet-based methods require multiple GPU days per dataset. Large Language Model (LLM)-driven approaches avoid training a large supernet and can provide quick feedback, but we observe an exploration bias: the LLM repeatedly proposes neural network designs within limited search space and fails to discover architectures across different latency ranges in the entire search space. To address this issue, we propose LLM-NAS: an LLM-driven Neural Architecture Search that can generate neural networks with high accuracy and low latency with reduced search cost. Our proposed LLM-NAS has three key components: 1) a complexity-driven partitioning engine that divides the search space by complexity to enforce diversity and mitigate exploration bias; 2) an LLM-powered architecture prompt co-evolution operator, in which the LLM first updates a knowledge base of design heuristics based on results from the previous round, then performs a guided evolution algorithm on architectures with prompts that incorporate this knowledge base. Prompts and designs improve together across rounds which avoids random guesswork and improve efficiency; 3) a zero-cost predictor to avoid training a large number of candidates from scratch. Experimental results show that on HW-NAS-Bench, LLM-NAS can achieve overall higher HV, lower IGD, and up to 54% lower latency than baselines at similar accuracy. Meanwhile, the search cost drops from days to minutes compared with traditional supernet baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15414v2">MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Developing large language models (LLMs) to cooperate and compete effectively within multi-agent systems (MASs) is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARSHAL, an end-to-end RL framework that incentivizes Multi-Agent Reasoning through Self-play witH strAtegic LLMs in both cooperative and competitive games. MARSHAL features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, MARSHAL agent trained from Qwen3-4B develops strong strategic abilities that generalize to held-out games with up to 28.7% performance improvements. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of MASs in reasoning benchmarks. When integrated into leading MASs, our MARSHAL agent achieves significant performance gains of up to 10.0% on AIME, 6.6% on GPQA-Diamond, and 3.5% on average across all benchmarks. These results establish end-to-end RL training with self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06942v3">HLPD: Aligning LLMs to Human Language Preference for Machine-Revised Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 20 pages, 10 figures, accepted by AAAI'26
    </div>
    <details class="paper-abstract">
      To prevent misinformation and social issues arising from trustworthy-looking content generated by LLMs, it is crucial to develop efficient and reliable methods for identifying the source of texts. Previous approaches have demonstrated exceptional performance in detecting texts fully generated by LLMs. However, these methods struggle when confronting more advanced LLM output or text with adversarial multi-task machine revision, especially in the black-box setting, where the generating model is unknown. To address this challenge, grounded in the hypothesis that human writing possesses distinctive stylistic patterns, we propose Human Language Preference Detection (HLPD). HLPD employs a reward-based alignment process, Human Language Preference Optimization (HLPO), to shift the scoring model's token distribution toward human-like writing, making the model more sensitive to human writing, therefore enhancing the identification of machine-revised text. We test HLPD in an adversarial multi-task evaluation framework that leverages a five-dimensional prompt generator and multiple advanced LLMs to create diverse revision scenarios. When detecting texts revised by GPT-series models, HLPD achieves a 15.11% relative improvement in AUROC over ImBD, surpassing Fast-DetectGPT by 45.56%. When evaluated on texts generated by advanced LLMs, HLPD achieves the highest average AUROC, exceeding ImBD by 5.53% and Fast-DetectGPT by 34.14%. Code will be made available at https://github.com/dfq2021/HLPD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.13068v3">Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02914v1">Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning techniques have substantially improved the performance of large language models (LLMs), raising expectations for their ability to provide accurate, truthful, and reliable information. However, emerging evidence suggests that iterative reasoning may foster belief entrenchment and confirmation bias, rather than enhancing truth-seeking behavior. In this study, we propose a systematic evaluation framework for belief entrenchment in LLM reasoning by leveraging the Martingale property from Bayesian statistics. This property implies that, under rational belief updating, the expected value of future beliefs should remain equal to the current belief, i.e., belief updates are unpredictable from the current belief. We propose the unsupervised, regression-based Martingale Score to measure violations of this property, which signal deviation from the Bayesian ability of updating on new evidence. In open-ended problem domains including event forecasting, value-laden questions, and academic paper review, we find such violations to be widespread across models and setups, where the current belief positively predicts future belief updates, a phenomenon which we term belief entrenchment. We identify the models, reasoning techniques, and domains more prone to belief entrenchment. Finally, we validate the Martingale Score by showing that it predicts ground-truth accuracy on problem domains where ground truth labels are available. This indicates that, while designed as an unsupervised metric that operates even in domains without access to ground truth, the Martingale Score is a useful proxy of the truth-seeking ability of a reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16873v2">Multimodal LLMs See Sentiment</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Understanding how visual content communicates sentiment is critical in an era where online interaction is increasingly dominated by this kind of media on social platforms. However, this remains a challenging problem, as sentiment perception is closely tied to complex, scene-level semantics. In this paper, we propose an original framework, MLLMsent, to investigate the sentiment reasoning capabilities of Multimodal Large Language Models (MLLMs) through three perspectives: (1) using those MLLMs for direct sentiment classification from images; (2) associating them with pre-trained LLMs for sentiment analysis on automatically generated image descriptions; and (3) fine-tuning the LLMs on sentiment-labeled image descriptions. Experiments on a recent and established benchmark demonstrate that our proposal, particularly the fine-tuned approach, achieves state-of-the-art results outperforming Lexicon-, CNN-, and Transformer-based baselines by up to 30.9%, 64.8%, and 42.4%, respectively, across different levels of evaluators' agreement and sentiment polarity categories. Remarkably, in a cross-dataset test, without any training on these new data, our model still outperforms, by up to 8.26%, the best runner-up, which has been trained directly on them. These results highlight the potential of the proposed visual reasoning scheme for advancing affective computing, while also establishing new benchmarks for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02910v1">In Silico Development of Psychometric Scales: Feasibility of Representative Population Data Simulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Developing and validating psychometric scales requires large samples, multiple testing phases, and substantial resources. Recent advances in Large Language Models (LLMs) enable the generation of synthetic participant data by prompting models to answer items while impersonating individuals of specific demographic profiles, potentially allowing in silico piloting before real data collection. Across four preregistered studies (N = circa 300 each), we tested whether LLM-simulated datasets can reproduce the latent structures and measurement properties of human responses. In Studies 1-2, we compared LLM-generated data with real datasets for two validated scales; in Studies 3-4, we created new scales using EFA on simulated data and then examined whether these structures generalized to newly collected human samples. Simulated datasets replicated the intended factor structures in three of four studies and showed consistent configural and metric invariance, with scalar invariance achieved for the two newly developed scales. However, correlation-based tests revealed substantial differences between real and synthetic datasets, and notable discrepancies appeared in score distributions and variances. Thus, while LLMs capture group-level latent structures, they do not approximate individual-level data properties. Simulated datasets also showed full internal invariance across gender. Overall, LLM-generated data appear useful for early-stage, group-level psychometric prototyping, but not as substitutes for individual-level validation. We discuss methodological limitations, risks of bias and data pollution, and ethical considerations related to in silico psychometric simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09481v2">Evaluating LLMs on Sequential API Call Through Automated Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs.We make our framework and benchmark publicly available to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12726v5">DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, as well as guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling the automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to generate reasoning questions with controllable question types and difficulty levels. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, by applying SFT on the base versions of these models using only our data, we even surpass their official final models that have undergone the full post-training process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.15230v4">PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 32 pages, 7 figures, 24 tables
    </div>
    <details class="paper-abstract">
      Neural Networks can be effectively compressed through pruning, significantly reducing storage and compute demands while maintaining predictive performance. Simple yet effective methods like magnitude pruning remove less important parameters and typically require a costly retraining procedure to restore performance. However, with the rise of LLMs, full retraining has become infeasible due to memory and compute constraints. This study challenges the practice of retraining all parameters by showing that updating a small subset of highly expressive parameters can suffice to recover or even enhance performance after pruning. Surprisingly, retraining just 0.01%-0.05% of the parameters in GPT-architectures can match the performance of full retraining across various sparsity levels, significantly reducing compute and memory requirements, and enabling retraining of models with up to 30 billion parameters on a single GPU in minutes. To bridge the gap to full retraining in the high sparsity regime, we introduce two novel LoRA variants that, unlike standard LoRA, allow merging adapters back without compromising sparsity. Going a step further, we show that these methods can be applied for memory-efficient layer-wise reconstruction, significantly enhancing state-of-the-art retraining-free methods like Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023). Our findings present a promising alternative to avoiding retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01830v2">OpenREAD: Reinforced Open-Ended Reasoning for End-to-End Autonomous Driving with LLM-as-Critic</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02841v1">Cross-Lingual Prompt Steerability: Towards Accurate and Robust LLM Behavior across Languages</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      System prompts provide a lightweight yet powerful mechanism for conditioning large language models (LLMs) at inference time. While prior work has focused on English-only settings, real-world deployments benefit from having a single prompt to operate reliably across languages. This paper presents a comprehensive study of how different system prompts steer models toward accurate and robust cross-lingual behavior. We propose a unified four-dimensional evaluation framework to assess system prompts in multilingual environments. Through large-scale experiments on five languages, three LLMs, and three benchmarks, we uncover that certain prompt components, such as CoT, emotion, and scenario, correlate with robust multilingual behavior. We develop a prompt optimization framework for multilingual settings and show it can automatically discover prompts that improve all metrics by 5-10%. Finally, we analyze over 10 million reasoning units and find that more performant system prompts induce more structured and consistent reasoning patterns, while reducing unnecessary language-switching. Together, we highlight system prompt optimization as a scalable path to accurate and robust multilingual LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02810v1">Phase-Adaptive LLM Framework with Multi-Stage Validation for Construction Robot Task Allocation: A Systematic Benchmark Against Traditional Optimization Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Multi-robot task allocation in construction automation has traditionally relied on optimization methods such as Dynamic Programming and Reinforcement Learning. This research introduces the LangGraph-based Task Allocation Agent (LTAA), an LLM-driven framework that integrates phase-adaptive allocation strategies, multi-stage validation with hierarchical retries, and dynamic prompting for efficient robot coordination. Although recent LLM approaches show potential for construction robotics, they largely lack rigorous validation and benchmarking against established algorithms. This paper presents the first systematic comparison of LLM-based task allocation with traditional methods in construction scenarios.The study validates LLM feasibility through SMART-LLM replication and addresses implementation challenges using a Self-Corrective Agent Architecture. LTAA leverages natural-language reasoning combined with structured validation mechanisms, achieving major computational gains reducing token usage by 94.6% and allocation time by 86% through dynamic prompting. The framework adjusts its strategy across phases: emphasizing execution feasibility early and workload balance in later allocations.The authors evaluate LTAA against Dynamic Programming, Q-learning, and Deep Q-Network (DQN) baselines using construction operations from the TEACh human-robot collaboration dataset. In the Heavy Excels setting, where robots have strong task specializations, LTAA achieves 77% task completion with superior workload balance, outperforming all traditional methods. These findings show that LLM-based reasoning with structured validation can match established optimization algorithms while offering additional advantages such as interpretability, adaptability, and the ability to update task logic without retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11306v2">iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Accepted in AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02763v1">SurveyEval: Towards Comprehensive Evaluation of LLM-Generated Academic Surveys</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      LLM-based automatic survey systems are transforming how users acquire information from the web by integrating retrieval, organization, and content synthesis into end-to-end generation pipelines. While recent works focus on developing new generation pipelines, how to evaluate such complex systems remains a significant challenge. To this end, we introduce SurveyEval, a comprehensive benchmark that evaluates automatically generated surveys across three dimensions: overall quality, outline coherence, and reference accuracy. We extend the evaluation across 7 subjects and augment the LLM-as-a-Judge framework with human references to strengthen evaluation-human alignment. Evaluation results show that while general long-text or paper-writing systems tend to produce lower-quality surveys, specialized survey-generation systems are able to deliver substantially higher-quality results. We envision SurveyEval as a scalable testbed to understand and improve automatic survey systems across diverse subjects and evaluation criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21830v3">GAPO: Robust Advantage Estimation for Real-World Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods like GRPO are popular for their critic-free, normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable outliers, leading to distorted advantage computation and increased noise. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively finds an outlier-free highest-density interval (HDI) per prompt and then uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation. This adaptive Q robustly handles skewed distributions while remaining plug-and-play and efficient. We validate GAPO on nine instruction-tuned LLMs (3B-14B) using a large internal dataset of 51,844 real-world, history-aware code-editing tasks across 10 languages, demonstrating consistent improvements in exact match accuracy over GRPO and its variant DAPO. Code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25045v2">Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods either focus on input-oriented feature extraction, such as supervised probes and Sparse Autoencoders (SAEs), or on output distribution inspection, such as logit-oriented approaches. A full understanding of LLM vector spaces, however, requires integrating both perspectives, something existing approaches struggle with due to constraints on latent feature definitions. We introduce the Hyperdimensional Probe, a hybrid supervised probe that combines symbolic representations with neural probing. Leveraging Vector Symbolic Architectures (VSAs) and hypervector algebra, it unifies prior methods: the top-down interpretability of supervised probes, SAE's sparsity-driven proxy space, and output-oriented logit investigation. This allows deeper input-focused feature extraction while supporting output-oriented investigation. Our experiments show that our method consistently extracts meaningful concepts across LLMs, embedding sizes, and setups, uncovering concept-driven patterns in analogy-oriented inference and QA-focused text generation. By supporting joint input-output analysis, this work advances semantic understanding of neural representations while unifying the complementary perspectives of prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02726v1">AuditCopilot: Leveraging LLMs for Fraud Detection in Double-Entry Bookkeeping</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Auditors rely on Journal Entry Tests (JETs) to detect anomalies in tax-related ledger records, but rule-based methods generate overwhelming false positives and struggle with subtle irregularities. We investigate whether large language models (LLMs) can serve as anomaly detectors in double-entry bookkeeping. Benchmarking SoTA LLMs such as LLaMA and Gemma on both synthetic and real-world anonymized ledgers, we compare them against JETs and machine learning baselines. Our results show that LLMs consistently outperform traditional rule-based JETs and classical ML baselines, while also providing natural-language explanations that enhance interpretability. These results highlight the potential of \textbf{AI-augmented auditing}, where human auditors collaborate with foundation models to strengthen financial integrity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02719v1">Emergent Bayesian Behaviour and Optimal Cue Combination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at explicit reasoning, but their implicit computational strategies remain underexplored. Decades of psychophysics research show that humans intuitively process and integrate noisy signals using near-optimal Bayesian strategies in perceptual tasks. We ask whether LLMs exhibit similar behaviour and perform optimal multimodal integration without explicit training or instruction. Adopting the psychophysics paradigm, we infer computational principles of LLMs from systematic behavioural studies. We introduce a behavioural benchmark - BayesBench: four magnitude estimation tasks (length, location, distance, and duration) over text and image, inspired by classic psychophysics, and evaluate a diverse set of nine LLMs alongside human judgments for calibration. Through controlled ablations of noise, context, and instruction prompts, we measure performance, behaviour and efficiency in multimodal cue-combination. Beyond accuracy and efficiency metrics, we introduce a Bayesian Consistency Score that detects Bayes-consistent behavioural shifts even when accuracy saturates. Our results show that while capable models often adapt in Bayes-consistent ways, accuracy does not guarantee robustness. Notably, GPT-5 Mini achieves perfect text accuracy but fails to integrate visual cues efficiently. This reveals a critical dissociation between capability and strategy, suggesting accuracy-centric benchmarks may over-index on performance while missing brittle uncertainty handling. These findings reveal emergent principled handling of uncertainty and highlight the correlation between accuracy and Bayesian tendencies. We release our psychophysics benchmark and consistency metric (https://bayes-bench.github.io) as evaluation tools and to inform future multimodal architecture designs.
    </details>
</div>
