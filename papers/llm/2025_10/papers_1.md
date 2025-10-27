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
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23579v2">BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 28 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Unlocking deep and interpretable biological reasoning from complex genomic data remains a major AI challenge limiting scientific progress. While current DNA foundation models excel at representing sequences, they struggle with multi-step reasoning and lack transparent, biologically meaningful explanations. BioReason addresses this by tightly integrating a DNA foundation model with a large language model (LLM), enabling the LLM to directly interpret and reason over genomic information. Through supervised fine-tuning and reinforcement learning, BioReason learns to produce logical, biologically coherent deductions. It achieves major performance gains, boosting KEGG-based disease pathway prediction accuracy from 86% to 98% and improving variant effect prediction by an average of 15% over strong baselines. BioReason can reason over unseen biological entities and explain its decisions step by step, offering a transformative framework for interpretable, mechanistic AI in biology. All data, code, and checkpoints are available at https://github.com/bowang-lab/BioReason
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21631v1">Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Knowledge distillation is a promising approach to transfer capabilities from complex teacher models to smaller, resource-efficient student models that can be deployed easily, particularly in task-aware scenarios. However, existing methods of task-aware distillation typically require substantial quantities of data which may be unavailable or expensive to obtain in many practical scenarios. In this paper, we address this challenge by introducing a novel strategy called Counterfactual-explanation-infused Distillation CoD for few-shot task-aware knowledge distillation by systematically infusing counterfactual explanations. Counterfactual explanations (CFEs) refer to inputs that can flip the output prediction of the teacher model with minimum perturbation. Our strategy CoD leverages these CFEs to precisely map the teacher's decision boundary with significantly fewer samples. We provide theoretical guarantees for motivating the role of CFEs in distillation, from both statistical and geometric perspectives. We mathematically show that CFEs can improve parameter estimation by providing more informative examples near the teacher's decision boundary. We also derive geometric insights on how CFEs effectively act as knowledge probes, helping the students mimic the teacher's decision boundaries more effectively than standard data. We perform experiments across various datasets and LLMs to show that CoD outperforms standard distillation approaches in few-shot regimes (as low as 8-512 samples). Notably, CoD only uses half of the original samples used by the baselines, paired with their corresponding CFEs and still improves performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23794v2">R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) integrates external knowledge with Large Language Models (LLMs) to enhance factual correctness and mitigate hallucination. However, dense retrievers often become the bottleneck of RAG systems due to their limited parameters compared to LLMs and their inability to perform step-by-step reasoning. While prompt-based iterative RAG attempts to address these limitations, it is constrained by human-designed workflows. To address these limitations, we propose $\textbf{R3-RAG}$, which uses $\textbf{R}$einforcement learning to make the LLM learn how to $\textbf{R}$eason and $\textbf{R}$etrieve step by step, thus retrieving comprehensive external knowledge and leading to correct answers. R3-RAG is divided into two stages. We first use cold start to make the model learn the manner of iteratively interleaving reasoning and retrieval. Then we use reinforcement learning to further harness its ability to better explore the external retrieval environment. Specifically, we propose two rewards for R3-RAG: 1) answer correctness for outcome reward, which judges whether the trajectory leads to a correct answer; 2) relevance-based document verification for process reward, encouraging the model to retrieve documents that are relevant to the user question, through which we can let the model learn how to iteratively reason and retrieve relevant documents to get the correct answer. Experimental results show that R3-RAG significantly outperforms baselines and can transfer well to different retrievers. We release R3-RAG at https://github.com/Yuan-Li-FNLP/R3-RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01930v3">Robust LLM Alignment via Distributionally Robust Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      A major challenge in aligning large language models (LLMs) with human preferences is the issue of distribution shift. LLM alignment algorithms rely on static preference datasets, assuming that they accurately represent real-world user preferences. However, user preferences vary significantly across geographical regions, demographics, linguistic patterns, and evolving cultural trends. This preference distribution shift leads to catastrophic alignment failures in many real-world applications. We address this problem using the principled framework of distributionally robust optimization, and develop two novel distributionally robust direct preference optimization (DPO) algorithms, namely, Wasserstein DPO (WDPO) and Kullback-Leibler DPO (KLDPO). We characterize the sample complexity of learning the optimal policy parameters for WDPO and KLDPO. Moreover, we propose scalable gradient descent-style learning algorithms by developing suitable approximations for the challenging minimax loss functions of WDPO and KLDPO. Our empirical experiments using benchmark data sets and LLMs demonstrate the superior performance of WDPO and KLDPO in substantially improving the alignment when there is a preference distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17853v2">CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 https://kathcym.github.io/CiteGuard_Page
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising assistants for scientific writing. However, there have been concerns regarding the quality and reliability of the generated text, one of which is the citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which is assessing whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves the prior baseline by 12.3%, and achieves up to 65.4% accuracy on the CiteME benchmark, on par with human-level performance (69.7%). It also enables the identification of alternative but valid citations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21561v1">Are the LLMs Capable of Maintaining at Least the Language Genus?</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display notable variation in multilingual behavior, yet the role of genealogical language structure in shaping this variation remains underexplored. In this paper, we investigate whether LLMs exhibit sensitivity to linguistic genera by extending prior analyses on the MultiQ dataset. We first check if models prefer to switch to genealogically related languages when prompt language fidelity is not maintained. Next, we investigate whether knowledge consistency is better preserved within than across genera. We show that genus-level effects are present but strongly conditioned by training resource availability. We further observe distinct multilingual strategies across LLMs families. Our findings suggest that LLMs encode aspects of genus-level structure, but training data imbalances remain the primary factor shaping their multilingual performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21557v1">Co-Sight: Enhancing LLM-Based Agents via Conflict-Aware Meta-Verification and Trustworthy Reasoning with Structured Facts</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Long-horizon reasoning in LLM-based agents often fails not from generative weakness but from insufficient verification of intermediate reasoning. Co-Sight addresses this challenge by turning reasoning into a falsifiable and auditable process through two complementary mechanisms: Conflict-Aware Meta-Verification (CAMV) and Trustworthy Reasoning with Structured Facts (TRSF). CAMV reformulates verification as conflict identification and targeted falsification, allocating computation only to disagreement hotspots among expert agents rather than to full reasoning chains. This bounds verification cost to the number of inconsistencies and improves efficiency and reliability. TRSF continuously organizes, validates, and synchronizes evidence across agents through a structured facts module. By maintaining verified, traceable, and auditable knowledge, it ensures that all reasoning is grounded in consistent, source-verified information and supports transparent verification throughout the reasoning process. Together, TRSF and CAMV form a closed verification loop, where TRSF supplies structured facts and CAMV selectively falsifies or reinforces them, yielding transparent and trustworthy reasoning. Empirically, Co-Sight achieves state-of-the-art accuracy on GAIA (84.4%) and Humanity's Last Exam (35.5%), and strong results on Chinese-SimpleQA (93.8%). Ablation studies confirm that the synergy between structured factual grounding and conflict-aware verification drives these improvements. Co-Sight thus offers a scalable paradigm for reliable long-horizon reasoning in LLM-based agents. Code is available at https://github.com/ZTE-AICloud/Co-Sight/tree/cosight2.0_benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08221v2">Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 26 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20075v2">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 21 pages, main paper 9 pages
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19225v2">RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become essential for unlocking advanced reasoning capabilities in large language models (LLMs). RL workflows involve interleaving rollout and training stages with fundamentally different resource requirements. Rollout typically dominates overall execution time, yet scales efficiently through multiple independent instances. In contrast, training requires tightly-coupled GPUs with full-mesh communication. Existing RL frameworks fall into two categories: co-located and disaggregated architectures. Co-located ones fail to address this resource tension by forcing both stages to share the same GPUs. Disaggregated architectures, without modifications of well-established RL algorithms, suffer from resource under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances on public clouds and spare capacity in production clusters, present significant cost-saving opportunities for accelerating RL workflows, if efficiently harvested for rollout. In this paper, we present RLBoost, a systematic solution for cost-efficient RL training that harvests preemptible GPU resources. Our key insight is that rollout's stateless and embarrassingly parallel nature aligns perfectly with preemptible and often fragmented resources. To efficiently utilize these resources despite frequent and unpredictable availability changes, RLBoost adopts a hybrid architecture with three key techniques: (1) adaptive rollout offload to dynamically adjust workloads on the reserved (on-demand) cluster, (2) pull-based weight transfer that quickly provisions newly available instances, and (3) token-level response collection and migration for efficient preemption handling and continuous load balancing. Extensive experiments show RLBoost increases training throughput by 1.51x-1.97x while improving cost efficiency by 28%-49% compared to using only on-demand GPU resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21524v1">EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 Accepted at the Workshop on Regulatable ML at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as agents in various contexts by providing tools at their disposal. However, LLM agents can exhibit unpredictable behaviors, including taking undesirable and/or unsafe actions. In order to measure the latent propensity of LLM agents for taking illegal actions under an EU legislative context, we introduce EU-Agent-Bench, a verifiable human-curated benchmark that evaluates an agent's alignment with EU legal norms in situations where benign user inputs could lead to unlawful actions. Our benchmark spans scenarios across several categories, including data protection, bias/discrimination, and scientific integrity, with each user request allowing for both compliant and non-compliant execution of the requested actions. Comparing the model's function calls against a rubric exhaustively supported by citations of the relevant legislature, we evaluate the legal compliance of frontier LLMs, and furthermore investigate the compliance effect of providing the relevant legislative excerpts in the agent's system prompt along with explicit instructions to comply. We release a public preview set for the research community, while holding out a private test set to prevent data contamination in evaluating upcoming models. We encourage future work extending agentic safety benchmarks to different legal jurisdictions and to multi-turn and multilingual interactions. We release our code on \href{https://github.com/ilijalichkovski/eu-agent-bench}{this URL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04103v2">How to Train Your LLM Web Agent: A Statistical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21513v1">Wisdom and Delusion of LLM Ensembles for Code Generation and Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems. To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool. We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17196v2">Shape it Up! Restoring LLM Safety during Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Finetuning large language models (LLMs) enables user-specific customization but introduces critical safety risks: even a few harmful examples can compromise safety alignment. A common mitigation strategy is to update the model more strongly on examples deemed safe, while downweighting or excluding those flagged as unsafe. However, because safety context can shift within a single example, updating the model equally on both harmful and harmless parts of a response is suboptimal-a coarse treatment we term static safety shaping. In contrast, we propose dynamic safety shaping (DSS), a framework that uses fine-grained safety signals to reinforce learning from safe segments of a response while suppressing unsafe content. To enable such fine-grained control during finetuning, we introduce a key insight: guardrail models, traditionally used for filtering, can be repurposed to evaluate partial responses, tracking how safety risk evolves throughout the response, segment by segment. This leads to the Safety Trajectory Assessment of Response (STAR), a token-level signal that enables shaping to operate dynamically over the training sequence. Building on this, we present STAR-DSS, guided by STAR scores, that robustly mitigates finetuning risks and delivers substantial safety improvements across diverse threats, datasets, and model families-all without compromising capability on intended tasks. We encourage future safety research to build on dynamic shaping principles for stronger mitigation against evolving finetuning risks. Our code is publicly available at https://github.com/poloclub/star-dss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15144v2">HugAgent: Evaluating LLMs in Simulating Individual-Level Human Reasoning on Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
    </div>
    <details class="paper-abstract">
      Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01268v2">AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21459v1">SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 to be published in: The 3rd International Conference on Foundation and Large Language Models (FLLM2025), IEEE, 2025
    </div>
    <details class="paper-abstract">
      Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14828v2">Fundamental Limitations in Pointwise Defences of LLM Finetuning APIs</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21440v1">Redefining Retrieval Evaluation in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Traditional Information Retrieval (IR) metrics, such as nDCG, MAP, and MRR, assume that human users sequentially examine documents with diminishing attention to lower ranks. This assumption breaks down in Retrieval Augmented Generation (RAG) systems, where search results are consumed by Large Language Models (LLMs), which, unlike humans, process all retrieved documents as a whole rather than sequentially. Additionally, traditional IR metrics do not account for related but irrelevant documents that actively degrade generation quality, rather than merely being ignored. Due to these two major misalignments, namely human vs. machine position discount and human relevance vs. machine utility, classical IR metrics do not accurately predict RAG performance. We introduce a utility-based annotation schema that quantifies both the positive contribution of relevant passages and the negative impact of distracting ones. Building on this foundation, we propose UDCG (Utility and Distraction-aware Cumulative Gain), a metric using an LLM-oriented positional discount to directly optimize the correlation with the end-to-end answer accuracy. Experiments on five datasets and six LLMs demonstrate that UDCG improves correlation by up to 36% compared to traditional metrics. Our work provides a critical step toward aligning IR evaluation with LLM consumers and enables more reliable assessment of RAG components
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19092v2">Reinforced Latent Reasoning for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive reasoning capabilities in complex problem-solving tasks, sparking growing interest in their application to preference reasoning in recommendation systems. Existing methods typically rely on fine-tuning with explicit chain-of-thought (CoT) data. However, these methods face significant practical limitations due to (1) the difficulty of obtaining high-quality CoT data in recommendation and (2) the high inference latency caused by generating CoT reasoning. In this work, we explore an alternative approach that shifts from explicit CoT reasoning to compact, information-dense latent reasoning. This approach eliminates the need for explicit CoT generation and improves inference efficiency, as few latent tokens can effectively capture the entire reasoning process. Building on this idea, we propose \textit{\underline{R}einforced \underline{Latent} \underline{R}easoning for \underline{R}ecommendation} (LatentR$^3$), a novel end-to-end training framework that leverages reinforcement learning (RL) to optimize latent reasoning without relying on any CoT data. LatentR$^3$ adopts a two-stage training strategy: first, supervised fine-tuning to initialize the latent reasoning module, followed by pure RL training to encourage exploration through a rule-based reward design. Our RL implementation is based on a modified GRPO algorithm, which reduces computational overhead during training and introduces continuous reward signals for more efficient learning. Extensive experiments demonstrate that LatentR$^3$ enables effective latent reasoning without any direct supervision of the reasoning process, significantly improving performance when integrated with different LLM-based recommendation methods. Our codes are available at https://github.com/xuwenxinedu/R3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21401v1">FLAMES: Fine-tuning LLMs to Synthesize Invariants for Smart Contract Security</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Smart contract vulnerabilities cost billions of dollars annually, yet existing automated analysis tools fail to generate deployable defenses. We present FLAMES, a novel automated approach that synthesizes executable runtime guards as Solidity "require" statements to harden smart contracts against exploits. Unlike prior work that relies on vulnerability labels, symbolic analysis, or natural language specifications, FLAMES employs domain-adapted large language models trained through fill-in-the-middle supervised fine-tuning on real-world invariants extracted from 514,506 verified contracts. Our extensive evaluation across three dimensions demonstrates FLAMES's effectiveness: (1) Compilation: FLAMES achieves 96.7% compilability for synthesized invariant (2) Semantic Quality: on a curated test set of 5,000 challenging invariants, FLAMES produces exact or semantically equivalent matches to ground truth in 44.5% of cases; (3) Exploit Mitigation: FLAMES prevents 22 out of 108 real exploits (20.4%) while preserving contract functionality, and (4) FLAMES successfully blocks the real-world APEMAGA incident by synthesizing a pre-condition that mitigates the attack. FLAMES establishes that domain-adapted LLMs can automatically generate production-ready security defenses for smart contracts without requiring vulnerability detection, formal specifications, or human intervention. We release our code, model weights, datasets, and evaluation infrastructure to enable reproducible research in this critical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20815v1">Generative Reasoning Recommendation via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Despite their remarkable reasoning capabilities across diverse domains, large language models (LLMs) face fundamental challenges in natively functioning as generative reasoning recommendation models (GRRMs), where the intrinsic modeling gap between textual semantics and collaborative filtering signals, combined with the sparsity and stochasticity of user feedback, presents significant obstacles. This work explores how to build GRRMs by adapting pre-trained LLMs, which achieves a unified understanding-reasoning-prediction manner for recommendation tasks. We propose GREAM, an end-to-end framework that integrates three components: (i) Collaborative-Semantic Alignment, which fuses heterogeneous textual evidence to construct semantically consistent, discrete item indices and auxiliary alignment tasks that ground linguistic representations in interaction semantics; (ii) Reasoning Curriculum Activation, which builds a synthetic dataset with explicit Chain-of-Thought supervision and a curriculum that progresses through behavioral evidence extraction, latent preference modeling, intent inference, recommendation formulation, and denoised sequence rewriting; and (iii) Sparse-Regularized Group Policy Optimization (SRPO), which stabilizes post-training via Residual-Sensitive Verifiable Reward and Bonus-Calibrated Group Advantage Estimation, enabling end-to-end optimization under verifiable signals despite sparse successes. GREAM natively supports two complementary inference modes: Direct Sequence Recommendation for high-throughput, low-latency deployment, and Sequential Reasoning Recommendation that first emits an interpretable reasoning chain for causal transparency. Experiments on three datasets demonstrate consistent gains over strong baselines, providing a practical path toward verifiable-RL-driven LLM recommenders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20810v1">On the Detectability of LLM-Generated Text: What Exactly Is LLM-Generated Text?</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      With the widespread use of large language models (LLMs), many researchers have turned their attention to detecting text generated by them. However, there is no consistent or precise definition of their target, namely "LLM-generated text". Differences in usage scenarios and the diversity of LLMs further increase the difficulty of detection. What is commonly regarded as the detecting target usually represents only a subset of the text that LLMs can potentially produce. Human edits to LLM outputs, together with the subtle influences that LLMs exert on their users, are blurring the line between LLM-generated and human-written text. Existing benchmarks and evaluation approaches do not adequately address the various conditions in real-world detector applications. Hence, the numerical results of detectors are often misunderstood, and their significance is diminishing. Therefore, detectors remain useful under specific conditions, but their results should be interpreted only as references rather than decisive indicators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20800v1">Compress to Impress: Efficient LLM Adaptation Using a Single Gradient Step on 100 Samples</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Recently, Sharma et al. suggested a method called Layer-SElective-Rank reduction (LASER) which demonstrated that pruning high-order components of carefully chosen LLM's weight matrices can boost downstream accuracy -- without any gradient-based fine-tuning. Yet LASER's exhaustive, per-matrix search (each requiring full-dataset forward passes) makes it impractical for rapid deployment. We demonstrate that this overhead can be removed and find that: (i) Only a small, carefully chosen subset of matrices needs to be inspected -- eliminating the layer-by-layer sweep, (ii) The gradient of each matrix's singular values pinpoints which matrices merit reduction, (iii) Increasing the factorization search space by allowing matrices rows to cluster around multiple subspaces and then decomposing each cluster separately further reduces overfitting on the original training data and further lifts accuracy by up to 24.6 percentage points, and finally, (iv) we discover that evaluating on just 100 samples rather than the full training data -- both for computing the indicative gradients and for measuring the final accuracy -- suffices to further reduce the search time; we explain that as adaptation to downstream tasks is dominated by prompting style, not dataset size. As a result, we show that combining these findings yields a fast and robust adaptation algorithm for downstream tasks. Overall, with a single gradient step on 100 examples and a quick scan of the top candidate layers and factorization techniques, we can adapt LLMs to new datasets -- entirely without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20782v1">A Use-Case Specific Dataset for Measuring Dimensions of Responsible Performance in LLM-generated Text</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 24 pages with 3 figures, to appear in Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)
    </div>
    <details class="paper-abstract">
      Current methods for evaluating large language models (LLMs) typically focus on high-level tasks such as text generation, without targeting a particular AI application. This approach is not sufficient for evaluating LLMs for Responsible AI dimensions like fairness, since protected attributes that are highly relevant in one application may be less relevant in another. In this work, we construct a dataset that is driven by a real-world application (generate a plain-text product description, given a list of product features), parameterized by fairness attributes intersected with gendered adjectives and product categories, yielding a rich set of labeled prompts. We show how to use the data to identify quality, veracity, safety, and fairness gaps in LLMs, contributing a proposal for LLM evaluation paired with a concrete resource for the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20768v1">RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20743v1">Empathic Prompting: Non-Verbal Context Integration for Multimodal LLM Conversations</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      We present Empathic Prompting, a novel framework for multimodal human-AI interaction that enriches Large Language Model (LLM) conversations with implicit non-verbal context. The system integrates a commercial facial expression recognition service to capture users' emotional cues and embeds them as contextual signals during prompting. Unlike traditional multimodal interfaces, empathic prompting requires no explicit user control; instead, it unobtrusively augments textual input with affective information for conversational and smoothness alignment. The architecture is modular and scalable, allowing integration of additional non-verbal modules. We describe the system design, implemented through a locally deployed DeepSeek instance, and report a preliminary service and usability evaluation (N=5). Results show consistent integration of non-verbal input into coherent LLM outputs, with participants highlighting conversational fluidity. Beyond this proof of concept, empathic prompting points to applications in chatbot-mediated communication, particularly in domains like healthcare or education, where users' emotional signals are critical yet often opaque in verbal exchanges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20721v1">User Perceptions of Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have seen rapid adoption for tasks such as drafting emails, summarizing meetings, and answering health questions. In such uses, users may need to share private information (e.g., health records, contact details). To evaluate LLMs' ability to identify and redact such private information, prior work developed benchmarks (e.g., ConfAIde, PrivacyLens) with real-life scenarios. Using these benchmarks, researchers have found that LLMs sometimes fail to keep secrets private when responding to complex tasks (e.g., leaking employee salaries in meeting summaries). However, these evaluations rely on LLMs (proxy LLMs) to gauge compliance with privacy norms, overlooking real users' perceptions. Moreover, prior work primarily focused on the privacy-preservation quality of responses, without investigating nuanced differences in helpfulness. To understand how users perceive the privacy-preservation quality and helpfulness of LLM responses to privacy-sensitive scenarios, we conducted a user study with 94 participants using 90 scenarios from PrivacyLens. We found that, when evaluating identical responses to the same scenario, users showed low agreement with each other on the privacy-preservation quality and helpfulness of the LLM response. Further, we found high agreement among five proxy LLMs, while each individual LLM had low correlation with users' evaluations. These results indicate that the privacy and helpfulness of LLM responses are often specific to individuals, and proxy LLMs are poor estimates of how real users would perceive these responses in privacy-sensitive scenarios. Our results suggest the need to conduct user-centered studies on measuring LLMs' ability to help users while preserving privacy. Additionally, future research could investigate ways to improve the alignment between proxy LLMs and users for better estimation of users' perceived privacy and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17385v2">TabR1: Taming GRPO for tabular reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Tabular prediction has traditionally relied on gradient-boosted decision trees and specialized deep learning models, which excel within tasks but provide limited interpretability and weak transfer across tables. Reasoning large language models (LLMs) promise cross-task adaptability with trans- parent reasoning traces, yet their potential has not been fully realized for tabular data. This paper presents TabR1, the first reasoning LLM for tabular prediction with multi-step reasoning. At its core is Permutation Relative Policy Optimization (PRPO), a simple yet efficient reinforcement learning method that encodes column-permutation invariance as a structural prior. By construct- ing multiple label-preserving permutations per sample and estimating advantages both within and across permutations, PRPO transforms sparse rewards into dense learning signals and improves generalization. With limited supervision, PRPO activates the reasoning ability of LLMs for tabular prediction, enhancing few-shot and zero-shot performance as well as interpretability. Comprehensive experiments demonstrate that TabR1 achieves performance comparable to strong baselines under full-supervision fine-tuning. In the zero-shot setting, TabR1 approaches the performance of strong baselines under the 32-shot setting. Moreover, TabR1 (8B) substantially outperforms much larger LLMs across various tasks, achieving up to 53.17% improvement over DeepSeek-R1 (685B).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18631v3">ReDit: Reward Dithering for Improved LLM Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 34 pages, 19 figures
    </div>
    <details class="paper-abstract">
      DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23946v2">Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 NeurIPS 2025. Code is available at https://github.com/MITIBM-FastCoder/LessonL
    </div>
    <details class="paper-abstract">
      Recent studies show that LLMs possess different skills and specialize in different tasks. In fact, we observe that their varied performance occur in several levels of granularity. For example, in the code optimization task, code LLMs excel at different optimization categories and no one dominates others. This observation prompts the question of how one leverages multiple LLM agents to solve a coding problem without knowing their complementary strengths a priori. We argue that a team of agents can learn from each other's successes and failures so as to improve their own performance. Thus, a lesson is the knowledge produced by an agent and passed on to other agents in the collective solution process. We propose a lesson-based collaboration framework, design the lesson solicitation--banking--selection mechanism, and demonstrate that a team of small LLMs with lessons learned can outperform a much larger LLM and other multi-LLM collaboration methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20653v1">Finding the Sweet Spot: Trading Quality, Cost, and Speed During Inference-Time LLM Reflection</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to evolve, practitioners face increasing options for enhancing inference-time performance without model retraining, including budget tuning and multi-step techniques like self-reflection. While these methods improve output quality, they create complex trade-offs among accuracy, cost, and latency that remain poorly understood across different domains. This paper systematically compares self-reflection and budget tuning across mathematical reasoning and translation tasks. We evaluate prominent LLMs, including Anthropic Claude, Amazon Nova, and Mistral families, along with other models under varying reflection depths and compute budgets to derive Pareto optimal performance frontiers. Our analysis reveals substantial domain dependent variation in self-reflection effectiveness, with performance gains up to 220\% in mathematical reasoning. We further investigate how reflection round depth and feedback mechanism quality influence performance across model families. To validate our findings in a real-world setting, we deploy a self-reflection enhanced marketing content localisation system at Lounge by Zalando, where it shows market-dependent effectiveness, reinforcing the importance of domain specific evaluation when deploying these techniques. Our results provide actionable guidance for selecting optimal inference strategies given specific domains and resource constraints. We open source our self-reflection implementation for reproducibility at https://github.com/aws-samples/sample-genai-reflection-for-bedrock.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19507v2">Teaming LLMs to Detect and Mitigate Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Accepted to NeurIPS 2025 workshop on Reliable ML from Unreliable Data
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated state-of-the-art results in large language model (LLM) hallucination detection and mitigation through consistency-based approaches which involve aggregating multiple responses sampled from a single LLM for a given prompt. These approaches help offset limitations stemming from the imperfect data on which LLMs are trained, which includes biases and under-representation of information required at deployment time among other limitations which can lead to hallucinations. We show that extending these single-model consistency methods to combine responses from multiple LLMs with different training data, training schemes and model architectures can result in substantial further improvements in hallucination detection and mitigation capabilities beyond their single-model consistency counterparts. We evaluate this "consortium consistency" approach across many model teams from a pool of 15 LLMs and explore under what conditions it is beneficial to team together different LLMs in this manner. Further, we show that these performance improvements often come with reduced inference costs, offsetting a significant drawback with single-model consistency methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13837v4">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 30 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20612v1">Black Box Absorption: LLMs Undermining Innovative Ideas</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly adopted as critical tools for accelerating innovation. This paper identifies and formalizes a systemic risk inherent in this paradigm: \textbf{Black Box Absorption}. We define this as the process by which the opaque internal architectures of LLM platforms, often operated by large-scale service providers, can internalize, generalize, and repurpose novel concepts contributed by users during interaction. This mechanism threatens to undermine the foundational principles of innovation economics by creating severe informational and structural asymmetries between individual creators and platform operators, thereby jeopardizing the long-term sustainability of the innovation ecosystem. To analyze this challenge, we introduce two core concepts: the idea unit, representing the transportable functional logic of an innovation, and idea safety, a multidimensional standard for its protection. This paper analyzes the mechanisms of absorption and proposes a concrete governance and engineering agenda to mitigate these risks, ensuring that creator contributions remain traceable, controllable, and equitable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20603v1">What Defines Good Reasoning in LLMs? Dissecting Reasoning Steps with Multi-Aspect Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on final-answer correctness is the dominant paradigm. This approach, however, provides a coarse signal for model improvement and overlooks the quality of the underlying reasoning process. We argue that a more granular evaluation of reasoning offers a more effective path to building robust models. We decompose reasoning quality into two dimensions: relevance and coherence. Relevance measures if a step is grounded in the problem; coherence measures if it follows logically from prior steps. To measure these aspects reliably, we introduce causal stepwise evaluation (CaSE). This method assesses each reasoning step using only its preceding context, which avoids hindsight bias. We validate CaSE against human judgments on our new expert-annotated benchmarks, MRa-GSM8K and MRa-MATH. More importantly, we show that curating training data with CaSE-evaluated relevance and coherence directly improves final task performance. Our work provides a scalable framework for analyzing, debugging, and improving LLM reasoning, demonstrating the practical value of moving beyond validity checks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14536v2">Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17472v2">Certified Self-Consistency: Statistical Guarantees and Test-Time Training for Reliable Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Recent advances such as self-consistency and test-time reinforcement learning (TTRL) improve the reliability of large language models (LLMs) without additional supervision, yet their underlying mechanisms and statistical guarantees remain poorly understood. We present a unified framework for certifiable inference in LLMs, showing that majority voting provides a statistical certificate of self-consistency: under mild assumptions, the aggregated answer coincides with the mode of the model's terminal distribution with high probability. We derive finite-sample and anytime-valid concentration bounds that quantify this confidence, and introduce the Martingale Majority Certificate (MMC), a sequential stopping rule that adaptively determines when sufficient samples have been drawn. We further prove that label-free post-training methods such as TTRL implicitly sharpen the answer distribution by exponentially tilting it toward its mode, thereby reducing the number of samples required for certification. Building on this insight, we propose new post-training objectives that explicitly optimise this trade-off between sharpness and bias. Together, these results explain and connect two central test-time scaling strategies, self-consistency and TTRL, within a single statistical framework for label-free, certifiable reliability in reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14101v2">MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have inherent limitations of faithfulness and factuality, commonly referred to as hallucinations. Several benchmarks have been developed that provide a test bed for factuality evaluation within the context of English-centric datasets, while relying on supplementary informative context like web links or text passages but ignoring the available structured factual resources. To this end, Knowledge Graphs (KGs) have been identified as a useful aid for hallucination mitigation, as they provide a structured way to represent the facts about entities and their relations with minimal linguistic overhead. We bridge the lack of KG paths and multilinguality for factual language modeling within the existing hallucination evaluation benchmarks and propose a KG-based multilingual, multihop benchmark called MultiHal framed for generative text evaluation. As part of our data collection pipeline, we mined 140k KG-paths from open-domain KGs, from which we pruned noisy KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation shows an absolute scale improvement by approximately 0.12 to 0.36 points for the semantic similarity score, 0.16 to 0.36 for NLI entailment and 0.29 to 0.42 for hallucination detection in KG-RAG over vanilla QA across multiple languages and multiple models, demonstrating the potential of KG integration. We anticipate MultiHal will foster future research towards several graph-based hallucination mitigation and fact-checking tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20508v1">Assessing the Political Fairness of Multilingual LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      The political biases of Large Language Models (LLMs) are usually assessed by simulating their answers to English surveys. In this work, we propose an alternative framing of political biases, relying on principles of fairness in multilingual translation. We systematically compare the translation quality of speeches in the European Parliament (EP), observing systematic differences with majority parties from left, center, and right being better translated than outsider parties. This study is made possible by a new, 21-way multiparallel version of EuroParl, the parliamentary proceedings of the EP, which includes the political affiliations of each speaker. The dataset consists of 1.5M sentences for a total of 40M words and 249M characters. It covers three years, 1000+ speakers, 7 countries, 12 EU parties, 25 EU committees, and hundreds of national parties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16559v2">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19733v2">Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factorized hypernetwork framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to 26x fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20369v1">Ask a Strong LLM Judge when Your Reward Model is Uncertain</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 NeurIPS 2025, 18 pages
    </div>
    <details class="paper-abstract">
      Reward model (RM) plays a pivotal role in reinforcement learning with human feedback (RLHF) for aligning large language models (LLMs). However, classical RMs trained on human preferences are vulnerable to reward hacking and generalize poorly to out-of-distribution (OOD) inputs. By contrast, strong LLM judges equipped with reasoning capabilities demonstrate superior generalization, even without additional training, but incur significantly higher inference costs, limiting their applicability in online RLHF. In this work, we propose an uncertainty-based routing framework that efficiently complements a fast RM with a strong but costly LLM judge. Our approach formulates advantage estimation in policy gradient (PG) methods as pairwise preference classification, enabling principled uncertainty quantification to guide routing. Uncertain pairs are forwarded to the LLM judge, while confident ones are evaluated by the RM. Experiments on RM benchmarks demonstrate that our uncertainty-based routing strategy significantly outperforms random judge calling at the same cost, and downstream alignment results showcase its effectiveness in improving online RLHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v4">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01839v2">Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) transition from static tools to fully agentic systems, their potential for transforming social science research has become increasingly evident. This paper introduces a structured framework for understanding the diverse applications of LLM-based agents, ranging from simple data processors to complex, multi-agent systems capable of simulating emergent social dynamics. By mapping this developmental continuum across six levels, the paper clarifies the technical and methodological boundaries between different agentic architectures, providing a comprehensive overview of current capabilities and future potential. It highlights how lower-tier systems streamline conventional tasks like text classification and data annotation, while higher-tier systems enable novel forms of inquiry, including the study of group dynamics, norm formation, and large-scale social processes. However, these advancements also introduce significant challenges, including issues of reproducibility, ethical oversight, and the risk of emergent biases. The paper critically examines these concerns, emphasizing the need for robust validation protocols, interdisciplinary collaboration, and standardized evaluation metrics. It argues that while LLM-based agents hold transformative potential for the social sciences, realizing this promise will require careful, context-sensitive deployment and ongoing methodological refinement. The paper concludes with a call for future research that balances technical innovation with ethical responsibility, encouraging the development of agentic systems that not only replicate but also extend the frontiers of social science, offering new insights into the complexities of human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20345v1">LLM-empowered knowledge graph construction: A survey</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) have long served as a fundamental infrastructure for structured knowledge representation and reasoning. With the advent of Large Language Models (LLMs), the construction of KGs has entered a new paradigm-shifting from rule-based and statistical pipelines to language-driven and generative frameworks. This survey provides a comprehensive overview of recent progress in LLM-empowered knowledge graph construction, systematically analyzing how LLMs reshape the classical three-layered pipeline of ontology engineering, knowledge extraction, and knowledge fusion. We first revisit traditional KG methodologies to establish conceptual foundations, and then review emerging LLM-driven approaches from two complementary perspectives: schema-based paradigms, which emphasize structure, normalization, and consistency; and schema-free paradigms, which highlight flexibility, adaptability, and open discovery. Across each stage, we synthesize representative frameworks, analyze their technical mechanisms, and identify their limitations. Finally, the survey outlines key trends and future research directions, including KG-based reasoning for LLMs, dynamic knowledge memory for agentic systems, and multimodal KG construction. Through this systematic review, we aim to clarify the evolving interplay between LLMs and knowledge graphs, bridging symbolic knowledge engineering and neural semantic understanding toward the development of adaptive, explainable, and intelligent knowledge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25033v3">VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Few-shot learning (FSL) aims to recognize novel concepts from only a few labeled support samples. Recent studies enhance support features by incorporating additional semantic information or designing complex semantic fusion modules. However, they still suffer from hallucinating semantics that contradict the visual evidence due to the lack of grounding in actual instances, resulting in noisy guidance and costly corrections. To address these issues, we propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment. It mainly consists of Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). Specifically, the CIP conditions an LLM on both class names and support images to generate precise class descriptions iteratively in a single structured reasoning pass. These descriptions not only enrich the semantic understanding of novel classes but also enable the zero-shot synthesis of semantically consistent images. The descriptions and synthetic images act respectively as complementary textual and visual prompts, providing high-level class semantics and low-level intra-class diversity to compensate for limited support data. Furthermore, the CGA jointly aligns the fused textual, support, and synthetic visual representations by minimizing the kernelized volume of the 3-dimensional parallelotope they span. It captures global and nonlinear relationships among all representations, enabling structured and consistent multimodal integration. The proposed VT-FSL method establishes new state-of-the-art performance across ten diverse benchmarks, including standard, cross-domain, and fine-grained few-shot learning scenarios. Code is available at https://github.com/peacelwh/VT-FSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20279v1">ResearchGPT: Benchmarking and Training LLMs for End-to-End Computer Science Research Workflows</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) advance, the ultimate vision for their role in science is emerging: we could build an AI collaborator to effectively assist human beings throughout the entire scientific research process. We refer to this envisioned system as ResearchGPT. Given that scientific research progresses through multiple interdependent phases, achieving this vision requires rigorous benchmarks that evaluate the end-to-end workflow rather than isolated sub-tasks. To this end, we contribute CS-54k, a high-quality corpus of scientific Q&A pairs in computer science, built from 14k CC-licensed papers. It is constructed through a scalable, paper-grounded pipeline that combines retrieval-augmented generation (RAG) with multi-stage quality control to ensure factual grounding. From this unified corpus, we derive two complementary subsets: CS-4k, a carefully curated benchmark for evaluating AI's ability to assist scientific research, and CS-50k, a large-scale training dataset. Extensive experiments demonstrate that CS-4k stratifies state-of-the-art LLMs into distinct capability tiers. Open models trained on CS-50k with supervised training and reinforcement learning demonstrate substantial improvements. Even 7B-scale models, when properly trained, outperform many larger proprietary systems, such as GPT-4.1, GPT-4o, and Gemini 2.5 Pro. This indicates that making AI models better research assistants relies more on domain-aligned training with high-quality data than on pretraining scale or general benchmark performance. We release CS-4k and CS-50k in the hope of fostering AI systems as reliable collaborators in CS research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20272v1">Limits of PRM-Guided Tree Search for Mathematical Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      While chain-of-thought prompting with Best-of-N (BoN) selection has become popular for mathematical reasoning in large language models (LLMs), its linear structure fails to capture the branching and exploratory nature of complex problem-solving. In this work, we propose an adaptive algorithm to maximize process reward model (PRM) scores over the intractable action space, and investigate whether PRM-guided tree search can improve mathematical reasoning by exploring multiple partial solution paths. Across $23$ diverse mathematical problems using Qwen2.5-Math-7B-Instruct with its associated PRM as a case study, we find that: (1) PRM-guided tree search shows no statistically significant improvements over BoN despite higher costs, (2) Monte Carlo tree search and beam search outperform other PRM-guided tree search methods, (3) PRMs poorly approximate state values and their reliability degrades with reasoning depth, and (4) PRMs generalize poorly out of distribution. This underperformance stems from tree search's greater reliance on unreliable PRM scores, suggesting different reward modeling is necessary before tree search can effectively enhance mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20270v1">ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      The tendency to find and exploit "shortcuts" to complete tasks poses significant risks for reliable assessment and deployment of large language models (LLMs). For example, an LLM agent with access to unit tests may delete failing tests rather than fix the underlying bug. Such behavior undermines both the validity of benchmark results and the reliability of real-world LLM coding assistant deployments. To quantify, study, and mitigate such behavior, we introduce ImpossibleBench, a benchmark framework that systematically measures LLM agents' propensity to exploit test cases. ImpossibleBench creates "impossible" variants of tasks from existing benchmarks like LiveCodeBench and SWE-bench by introducing direct conflicts between the natural-language specification and the unit tests. We measure an agent's "cheating rate" as its pass rate on these impossible tasks, where any pass necessarily implies a specification-violating shortcut. As a practical framework, ImpossibleBench is not just an evaluation but a versatile tool. We demonstrate its utility for: (1) studying model behaviors, revealing more fine-grained details of cheating behaviors from simple test modification to complex operator overloading; (2) context engineering, showing how prompt, test access and feedback loop affect cheating rates; and (3) developing monitoring tools, providing a testbed with verified deceptive solutions. We hope ImpossibleBench serves as a useful framework for building more robust and reliable LLM systems. Our implementation can be found at https://github.com/safety-research/impossiblebench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15640v2">Multilingual LLM Prompting Strategies for Medical English-Vietnamese Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 This version has been withdrawn after receiving the conference review results. We are currently extending and reorganizing the work into a new study
    </div>
    <details class="paper-abstract">
      Medical English-Vietnamese machine translation (En-Vi MT) is essential for healthcare access and communication in Vietnam, yet Vietnamese remains a low-resource and under-studied language. We systematically evaluate prompting strategies for six multilingual LLMs (0.5B-9B parameters) on the MedEV dataset, comparing zero-shot, few-shot, and dictionary-augmented prompting with Meddict, an English-Vietnamese medical lexicon. Results show that model scale is the primary driver of performance: larger LLMs achieve strong zero-shot results, while few-shot prompting yields only marginal improvements. In contrast, terminology-aware cues and embedding-based example retrieval consistently improve domain-specific translation. These findings underscore both the promise and the current limitations of multilingual LLMs for medical En-Vi MT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17884v2">PhantomLint: Principled Detection of Hidden LLM Prompts in Structured Documents</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Hidden LLM prompts have appeared in online documents with increasing frequency. Their goal is to trigger indirect prompt injection attacks while remaining undetected from human oversight, to manipulate LLM-powered automated document processing systems, against applications as diverse as r\'esum\'e screeners through to academic peer review processes. Detecting hidden LLM prompts is therefore important for ensuring trust in AI-assisted human decision making. This paper presents the first principled approach to hidden LLM prompt detection in structured documents. We implement our approach in a prototype tool called PhantomLint. We evaluate PhantomLint against a corpus of 3,402 documents, including both PDF and HTML documents, and covering academic paper preprints, CVs, theses and more. We find that our approach is generally applicable against a wide range of methods for hiding LLM prompts from visual inspection, has a very low false positive rate (approx. 0.092%), is practically useful for detecting hidden LLM prompts in real documents, while achieving acceptable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20260v1">Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 RecSys 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) empower recommendation systems through their advanced reasoning and planning capabilities. However, the dynamic nature of user interests and content poses a significant challenge: While initial fine-tuning aligns LLMs with domain knowledge and user preferences, it fails to capture such real-time changes, necessitating robust update mechanisms. This paper investigates strategies for updating LLM-powered recommenders, focusing on the trade-offs between ongoing fine-tuning and Retrieval-Augmented Generation (RAG). Using an LLM-powered user interest exploration system as a case study, we perform a comparative analysis of these methods across dimensions like cost, agility, and knowledge incorporation. We propose a hybrid update strategy that leverages the long-term knowledge adaptation of periodic fine-tuning with the agility of low-cost RAG. We demonstrate through live A/B experiments on a billion-user platform that this hybrid approach yields statistically significant improvements in user satisfaction, offering a practical and cost-effective framework for maintaining high-quality LLM-powered recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19487v3">Evolution of Cooperation in LLM-Agent Societies: A Preliminary Study Using Different Punishment Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 20 pages, 10 figures, Accepted for presentation as a full paper at the COINE 2025 workshop at AAMAS 2025 (https://coin-workshop.github.io/coine-2025-detroit/accepted_for_presentation.html)
    </div>
    <details class="paper-abstract">
      The evolution of cooperation has been extensively studied using abstract mathematical models and simulations. Recent advances in Large Language Models (LLMs) and the rise of LLM agents have demonstrated their ability to perform social reasoning, thus providing an opportunity to test the emergence of norms in more realistic agent-based simulations with human-like reasoning using natural language. In this research, we investigate whether the cooperation dynamics presented in Boyd and Richerson's model persist in a more realistic simulation of the Diner's Dilemma using LLM agents compared to the abstract mathematical nature in the work of Boyd and Richerson. Our findings indicate that agents follow the strategies defined in the Boyd and Richerson model, and explicit punishment mechanisms drive norm emergence, reinforcing cooperative behaviour even when the agent strategy configuration varies. Our results suggest that LLM-based Multi-Agent System simulations, in fact, can replicate the evolution of cooperation predicted by the traditional mathematical models. Moreover, our simulations extend beyond the mathematical models by integrating natural language-driven reasoning and a pairwise imitation method for strategy adoption, making them a more realistic testbed for cooperative behaviour in MASs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09721v3">A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at https://github.com/lisaGuojl/LLM-Agent-SE-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20208v1">Decoding-Free Sampling Strategies for LLM Marginalization</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Modern language models operate on subword-tokenized text in order to make a trade-off between model size, inference speed, and vocabulary coverage. A side effect of this is that, during inference, models are evaluated by measuring the probability of only the specific tokenization produced as the output, despite there being many possible ways to represent the same text with a subword vocabulary. Recent studies have argued instead for evaluating LLMs by marginalization - the probability mass of all tokenizations of a given text. Marginalization is difficult due to the number of possible tokenizations of a text, so often approximate marginalization is done via sampling. However, a downside of sampling is that an expensive generation step must be performed by the LLM for each sample, which limits the number of samples that can be acquired given a runtime budget, and therefore also the accuracy of the approximation. Since computing the probability of a sequence given the tokenization is relatively cheap compared to actually generating it, we investigate sampling strategies that are decoding-free - they require no generation from the LLM, instead relying entirely on extremely cheap sampling strategies that are model and tokenizer agnostic. We investigate the approximation quality and speed of decoding-free sampling strategies for a number of open models to find that they provide sufficiently accurate marginal estimates at a small fraction of the runtime cost and demonstrate its use on a set of downstream inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v5">MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 39 pages, 11 figures. The paper is accepted at NeurIPS 2025 Datasets & Benchmarks Track, and the latest version adds modifications in camera-ready
    </div>
    <details class="paper-abstract">
      The ability to recognize patterns from examples and apply them to new ones is a primal ability for general intelligence, and is widely studied by psychology and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually <10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations often focus on classification, and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context reasoning benchmark for pattern recognition that asks LLM to predict output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for many-shot in-context reasoning, and acquired many insightful findings including scaling effect, robustness, inductive vs. transductive reasoning, retrieval Augmented Generation (RAG), coding for inductive reasoning, cross-domain generalizability, etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11821v2">Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      This paper investigates Reinforcement Learning (RL) approaches to enhance the reasoning capabilities of Large Language Model (LLM) agents in long-horizon, multi-turn scenarios. Although RL algorithms such as Group Relative Policy Optimization (GRPO) and Proximal Policy Optimization (PPO) have been widely applied to train multi-turn LLM agents, they typically rely only on sparse outcome rewards and lack dense intermediate signals across multiple decision steps, limiting their performance on complex reasoning tasks. To bridge this gap, we present the first systematic study of \textit{turn-level reward design} for multi-turn RL algorithms and agent applications. By integrating turn-level rewards, we extend GRPO and PPO to their respective multi-turn variants, enabling fine-grained credit assignment. We conduct case studies on multi-turn reasoning-augmented search agents, where we carefully design two types of turn-level rewards: verifiable and LLM-as-judge. Our experiments on multi-turn search tasks demonstrate that incorporating well-designed turn-level rewards enables RL algorithms to significantly outperform baseline methods with trajectory-level rewards. Both training and validation reward curves illustrate that our method achieves \textit{greater stability}, \textit{faster convergence}, and \textit{higher accuracy}. Numerical results across diverse question-answering datasets further show that our approach consistently delivers highest answer correctness and 100\% format correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16614v2">Count Counts: Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become a compelling way to strengthen the multi step reasoning ability of Large Language Models (LLMs). However, prevalent RL paradigms still lean on sparse outcome-based rewards and limited exploration, which often drives LLMs toward repetitive and suboptimal reasoning patterns. In this paper, we study the central question of how to design exploration for LLM reasoning and introduce MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards), a novel RL algorithm that augments policy optimization with a principled intrinsic reward. Building on the idea of count-based exploration, MERCI leverages a lightweight Coin Flipping Network (CFN) to estimate the pseudo count and further epistemic uncertainty over reasoning trajectories, and converts them into an intrinsic reward that values novelty while preserving the learning signal from task rewards. We integrate MERCI into some advanced RL frameworks like Group Relative Policy Optimization (GRPO). Experiments on complex reasoning benchmarks demonstrate that MERCI encourages richer and more varied chains of thought, significantly improves performance over strong baselines, and helps the policy escape local routines to discover better solutions. It indicates that our targeted intrinsic motivation can make exploration reliable for language model reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18470v2">CircuitSeer: Mining High-Quality Data by Probing Mathematical Reasoning Circuits in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities, but scaling their performance often relies on massive reasoning datasets that are computationally expensive to train on. Existing data selection methods aim to curate smaller, high-quality subsets but often rely on costly external models or opaque heuristics. In this work, we shift the focus from external heuristics to the model's internal mechanisms. We find that complex reasoning tasks consistently activate a sparse, specialized subset of attention heads, forming core reasoning circuits. Building on this insight, we propose CircuitSeer, a novel data selection method that quantifies the reasoning complexity of data by measuring its influence on these crucial circuits. Extensive experiments on 4 models and 9 datasets demonstrate CircuitSeer's superiority. Notably, fine-tuning Qwen2.5-Math-7B on just 10% of data selected by our method achieves a 1.4-point gain in average Pass@1 over training on the full dataset, highlighting its efficiency and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25271v4">RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02945v2">Quantitative LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge is a framework where a large language model (LLM) evaluates the output of another LLM. While LLMs excel at producing qualitative textual evaluations, they often struggle to predict human preferences and numeric scores. We propose quantitative LLM judges, which align evaluation scores of existing LLM judges to humans in a given domain using regression models. The models are trained to improve the score of the original judge using its rationale and score. We present four quantitative judges for different types of absolute and relative feedback, which showcases the generality and versatility of our framework. Our framework is more computationally efficient than supervised fine-tuning and can be more statistically efficient when human feedback is limited, which is expected in practice. We validate these claims empirically on four datasets using two base judges. Our experiments show that quantitative judges can improve the predictive power of existing judges through post-hoc modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v4">How Far Have LLMs Come Toward Automated SATD Taxonomy Construction?</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 5 pages, APSEC 2025
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. In this study, we investigated to what extent large language models (LLMs) could generate SATD taxonomies. We designed a structured, LLM-driven pipeline that mirrors the taxonomy construction steps researchers typically follow. We evaluated it on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovered domain-specific categories reported in prior work, such as Layer Configuration in machine learning. It also completed taxonomy generation in under two hours and for less than $1, even on the largest dataset. These results suggest that, while full automation remains challenging, LLMs can support semi-automated SATD taxonomy construction. Furthermore, our work opens up avenues for future work, such as automated taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20154v1">Are Stereotypes Leading LLMs' Zero-Shot Stance Detection ?</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Accepted in EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Large Language Models inherit stereotypes from their pretraining data, leading to biased behavior toward certain social groups in many Natural Language Processing tasks, such as hateful speech detection or sentiment analysis. Surprisingly, the evaluation of this kind of bias in stance detection methods has been largely overlooked by the community. Stance Detection involves labeling a statement as being against, in favor, or neutral towards a specific target and is among the most sensitive NLP tasks, as it often relates to political leanings. In this paper, we focus on the bias of Large Language Models when performing stance detection in a zero-shot setting. We automatically annotate posts in pre-existing stance detection datasets with two attributes: dialect or vernacular of a specific group and text complexity/readability, to investigate whether these attributes influence the model's stance detection decisions. Our results show that LLMs exhibit significant stereotypes in stance detection tasks, such as incorrectly associating pro-marijuana views with low text complexity and African American dialect with opposition to Donald Trump.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20150v1">Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at https://github.com/yaochenzhu/Rank-GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13992v2">AssistedDS: Benchmarking How External Domain Knowledge Assists LLMs in Automated Data Science</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced the automation of data science workflows. Yet it remains unclear whether they can critically leverage external domain knowledge as human data scientists do in practice. To answer this question, we introduce AssistedDS (Assisted Data Science), a benchmark designed to systematically evaluate how LLMs handle domain knowledge in tabular prediction tasks. AssistedDS features both synthetic datasets with explicitly known generative mechanisms and real-world Kaggle competitions, each accompanied by curated bundles of helpful and adversarial documents. These documents provide domain-specific insights into data cleaning, feature engineering, and model selection. We assess state-of-the-art LLMs on their ability to discern and apply beneficial versus harmful domain knowledge, evaluating submission validity, information recall, and predictive performance. Our results demonstrate three key findings: (1) LLMs frequently exhibit an uncritical adoption of provided information, significantly impairing their predictive performance when adversarial content is introduced, (2) helpful guidance is often insufficient to counteract the negative influence of adversarial information, and (3) in Kaggle datasets, LLMs often make errors in handling time-series data, applying consistent feature engineering across different folds, and interpreting categorical variables correctly. These findings highlight a substantial gap in current models' ability to critically evaluate and leverage expert knowledge, underscoring an essential research direction for developing more robust, knowledge-aware automated data science systems. Our data and code are publicly available here: https://github.com/jeremyxianx/Assisted-DS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20352v2">RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have shown outstanding potential for role-playing applications. Evaluating these capabilities is becoming crucial yet remains challenging. Existing benchmarks mostly adopt a \textbf{character-centric} approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications. To address this limitation, we introduce RMTBench, a comprehensive \textbf{user-centric} bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds. RMTBench includes custom characters with detailed backgrounds and abstract characters defined by simple traits, enabling evaluation across various user scenarios. Our benchmark constructs dialogues based on explicit user motivations rather than character descriptions, ensuring alignment with practical user applications. Furthermore, we construct an authentic multi-turn dialogue simulation mechanism. With carefully selected evaluation dimensions and LLM-based scoring, this mechanism captures the complex intention of conversations between the user and the character. By shifting focus from character background to user intention fulfillment, RMTBench bridges the gap between academic evaluation and practical deployment requirements, offering a more effective framework for assessing role-playing capabilities in LLMs. All code and datasets will be released soon. We release the datasets at https://huggingface.co/datasets/xiangh/RMTBENCH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18469v5">Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Accepted to NAACL 2025 Main (Oral)
    </div>
    <details class="paper-abstract">
      Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20039v1">Beyond One-Way Influence: Bidirectional Opinion Dynamics in Multi-Turn Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered chatbots are increasingly used for opinion exploration. Prior research examined how LLMs alter user views, yet little work extended beyond one-way influence to address how user input can affect LLM responses and how such bi-directional influence manifests throughout the multi-turn conversations. This study investigates this dynamic through 50 controversial-topic discussions with participants (N=266) across three conditions: static statements, standard chatbot, and personalized chatbot. Results show that human opinions barely shifted, while LLM outputs changed more substantially, narrowing the gap between human and LLM stance. Personalization amplified these shifts in both directions compared to the standard setting. Analysis of multi-turn conversations further revealed that exchanges involving participants' personal stories were most likely to trigger stance changes for both humans and LLMs. Our work highlights the risk of over-alignment in human-LLM interaction and the need for careful design of personalized chatbots to more thoughtfully and stably align with users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20036v1">ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Preprint under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents rely on external tools to solve complex tasks, but real-world toolsets often contain redundant tools with overlapping names and descriptions, introducing ambiguity and reducing selection accuracy. LLMs also face strict input context limits, preventing efficient consideration of large toolsets. To address these challenges, we propose ToolScope, which includes: (1) ToolScopeMerger with Auto-Correction to automatically audit and fix tool merges, reducing redundancy, and (2) ToolScopeRetriever to rank and select only the most relevant tools for each query, compressing toolsets to fit within context limits without sacrificing accuracy. Evaluations on three state-of-the-art LLMs and three open-source tool-use benchmarks show gains of 8.38% to 38.6% in tool selection accuracy, demonstrating ToolScope's effectiveness in enhancing LLM tool use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18882v3">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20001v1">Beyond MedQA: Towards Real-world Clinical Decision Making in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for clinical use. They are often evaluated using datasets such as MedQA. However, Many medical datasets, such as MedQA, rely on simplified Question-Answering (Q\A) that underrepresents real-world clinical decision-making. Based on this, we propose a unifying paradigm that characterizes clinical decision-making tasks along two dimensions: Clinical Backgrounds and Clinical Questions. As the background and questions approach the real clinical environment, the difficulty increases. We summarize the settings of existing datasets and benchmarks along two dimensions. Then we review methods to address clinical decision-making, including training-time and test-time techniques, and summarize when they help. Next, we extend evaluation beyond accuracy to include efficiency, explainability. Finally, we highlight open challenges. Our paradigm clarifies assumptions, standardizes comparisons, and guides the development of clinically meaningful LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08872v2">GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at https://github.com/ulab-uiuc/GTAlign .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19988v1">LLM-Augmented Symbolic NLU System for More Reliable Continuous Causal Statement Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 18 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Despite the broad applicability of large language models (LLMs), their reliance on probabilistic inference makes them vulnerable to errors such as hallucination in generated facts and inconsistent output structure in natural language understanding (NLU) tasks. By contrast, symbolic NLU systems provide interpretable understanding grounded in curated lexicons, semantic resources, and syntactic & semantic interpretation rules. They produce relational representations that can be used for accurate reasoning and planning, as well as incremental debuggable learning. However, symbolic NLU systems tend to be more limited in coverage than LLMs and require scarce knowledge representation and linguistics skills to extend and maintain. This paper explores a hybrid approach that integrates the broad-coverage language processing of LLMs with the symbolic NLU capabilities of producing structured relational representations to hopefully get the best of both approaches. We use LLMs for rephrasing and text simplification, to provide broad coverage, and as a source of information to fill in knowledge gaps more automatically. We use symbolic NLU to produce representations that can be used for reasoning and for incremental learning. We evaluate this approach on the task of extracting and interpreting quantities and causal laws from commonsense science texts, along with symbolic- and LLM-only pipelines. Our results suggest that our hybrid method works significantly better than the symbolic-only pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19986v1">Automating Iconclass: LLMs and RAG for Large-Scale Classification of Religious Woodcuts</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 29 pages, 7 figures. First presented at the "Digital Humanities and Artificial Intelligence" conference at the University of Reading on 17 June 2024
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for classifying early modern religious images by using Large Language Models (LLMs) and vector databases in combination with Retrieval-Augmented Generation (RAG). The approach leverages the full-page context of book illustrations from the Holy Roman Empire, allowing the LLM to generate detailed descriptions that incorporate both visual and textual elements. These descriptions are then matched to relevant Iconclass codes through a hybrid vector search. This method achieves 87% and 92% precision at five and four levels of classification, significantly outperforming traditional image and keyword-based searches. By employing full-page descriptions and RAG, the system enhances classification accuracy, offering a powerful tool for large-scale analysis of early modern visual archives. This interdisciplinary approach demonstrates the growing potential of LLMs and RAG in advancing research within art history and digital humanities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19886v1">An Expert-grounded benchmark of General Purpose LLMs in LCA</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Purpose: Artificial intelligence (AI), and in particular large language models (LLMs), are increasingly being explored as tools to support life cycle assessment (LCA). While demonstrations exist across environmental and social domains, systematic evidence on their reliability, robustness, and usability remains limited. This study provides the first expert-grounded benchmark of LLMs in LCA, addressing the absence of standardized evaluation frameworks in a field where no clear ground truth or consensus protocols exist. Methods: We evaluated eleven general-purpose LLMs, spanning both commercial and open-source families, across 22 LCA-related tasks. Seventeen experienced practitioners reviewed model outputs against criteria directly relevant to LCA practice, including scientific accuracy, explanation quality, robustness, verifiability, and adherence to instructions. We collected 168 expert reviews. Results: Experts judged 37% of responses to contain inaccurate or misleading information. Ratings of accuracy and quality of explanation were generally rated average or good on many models even smaller models, and format adherence was generally rated favourably. Hallucination rates varied significantly, with some models producing hallucinated citations at rates of up to 40%. There was no clear-cut distinction between ratings on open-weight versus closed-weight LLMs, with open-weight models outperforming or competing on par with closed-weight models on criteria such as accuracy and quality of explanation. Conclusion: These findings highlight the risks of applying LLMs na\"ively in LCA, such as when LLMs are treated as free-form oracles, while also showing benefits especially around quality of explanation and alleviating labour intensiveness of simple tasks. The use of general-purpose LLMs without grounding mechanisms presents ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19875v1">Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) scale to million-token contexts, traditional Mechanistic Interpretability techniques for analyzing attention scale quadratically with context length, demanding terabytes of memory beyond 100,000 tokens. We introduce Sparse Tracing, a novel technique that leverages dynamic sparse attention to efficiently analyze long context attention patterns. We present Stream, a compilable hierarchical pruning algorithm that estimates per-head sparse attention masks in near-linear time $O(T \log T)$ and linear space $O(T)$, enabling one-pass interpretability at scale. Stream performs a binary-search-style refinement to retain only the top-$k$ key blocks per query while preserving the model's next-token behavior. We apply Stream to long chain-of-thought reasoning traces and identify thought anchors while pruning 97-99\% of token interactions. On the RULER benchmark, Stream preserves critical retrieval paths while discarding 90-96\% of interactions and exposes layer-wise routes from the needle to output. Our method offers a practical drop-in tool for analyzing attention patterns and tracing information flow without terabytes of caches. By making long context interpretability feasible on consumer GPUs, Sparse Tracing helps democratize chain-of-thought monitoring. Code is available at https://anonymous.4open.science/r/stream-03B8/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19811v1">Hubble: a Model Suite to Advance the Study of LLM Memorization</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      We present Hubble, a suite of fully open-source large language models (LLMs) for the scientific study of LLM memorization. Hubble models come in standard and perturbed variants: standard models are pretrained on a large English corpus, and perturbed models are trained in the same way but with controlled insertion of text (e.g., book passages, biographies, and test sets) designed to emulate key memorization risks. Our core release includes 8 models -- standard and perturbed models with 1B or 8B parameters, pretrained on 100B or 500B tokens -- establishing that memorization risks are determined by the frequency of sensitive data relative to size of the training corpus (i.e., a password appearing once in a smaller corpus is memorized better than the same password in a larger corpus). Our release also includes 6 perturbed models with text inserted at different pretraining phases, showing that sensitive data without continued exposure can be forgotten. These findings suggest two best practices for addressing memorization risks: to dilute sensitive data by increasing the size of the training corpus, and to order sensitive data to appear earlier in training. Beyond these general empirical findings, Hubble enables a broad range of memorization research; for example, analyzing the biographies reveals how readily different types of private information are memorized. We also demonstrate that the randomized insertions in Hubble make it an ideal testbed for membership inference and machine unlearning, and invite the community to further explore, benchmark, and build upon our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19807v1">Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Code: https://github.com/dvlab-research/Scaf-GRPO
    </div>
    <details class="paper-abstract">
      Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19799v1">Integrating Transparent Models, LLMs, and Practitioner-in-the-Loop: A Case of Nonprofit Program Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Public and nonprofit organizations often hesitate to adopt AI tools because most models are opaque even though standard approaches typically analyze aggregate patterns rather than offering actionable, case-level guidance. This study tests a practitioner-in-the-loop workflow that pairs transparent decision-tree models with large language models (LLMs) to improve predictive accuracy, interpretability, and the generation of practical insights. Using data from an ongoing college-success program, we build interpretable decision trees to surface key predictors. We then provide each tree's structure to an LLM, enabling it to reproduce case-level predictions grounded in the transparent models. Practitioners participate throughout feature engineering, model design, explanation review, and usability assessment, ensuring that field expertise informs the analysis at every stage. Results show that integrating transparent models, LLMs, and practitioner input yields accurate, trustworthy, and actionable case-level evaluations, offering a viable pathway for responsible AI adoption in the public and nonprofit sectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19791v1">ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Tool calling has become increasingly popular for Large Language Models (LLMs). However, for large tool sets, the resulting tokens would exceed the LLM's context window limit, making it impossible to include every tool. Hence, an external retriever is used to provide LLMs with the most relevant tools for a query. Existing retrieval models rank tools based on the similarity between a user query and a tool description (TD). This leads to suboptimal retrieval as user requests are often poorly aligned with the language of TD. To remedy the issue, we propose ToolDreamer, a framework to condition retriever models to fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e., description of tools that the LLM feels will be potentially useful for the query. The framework enables a more natural alignment between queries and tools within the language space of TD's. We apply ToolDreamer on the ToolRet dataset and show that our method improves the performance of sparse and dense retrievers with and without training, thus showcasing its flexibility. Through our proposed framework, our aim is to offload a portion of the reasoning burden to the retriever so that the LLM may effectively handle a large collection of tools without inundating its context window.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19771v1">Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19747v1">Review of Tools for Zero-Code LLM Based Application Development</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted in 6th World Conference on Artificial Intelligence: Advances and Applications (WCAIAA 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming software creation by enabling zero code development platforms. Our survey reviews recent platforms that let users build applications without writing code, by leveraging LLMs as the brains of the development process. We adopt a broad survey methodology, categorizing platforms based on key dimensions such as interface style, backend integration, output type, and extensibility. We analyze both dedicated LLM based app builders (OpenAI's custom GPTs, Bolt.new, Dust.tt, Flowise, Cognosys) and general no code platforms (e.g., Bubble, Glide) that integrate LLM capabilities. We present a taxonomy categorizing these platforms by their interface (conversational, visual, etc.), supported LLM backends, output type (chatbot, full application, workflow), and degree of extensibility. Core features such as autonomous agents, memory management, workflow orchestration, and API integrations are in scope of the survey. We provide a detailed comparison, highlighting each platform's strengths and limitations. Trade offs (customizability, scalability, vendor lock-in) are discussed in comparison with traditional and low code development approaches. Finally, we outline future directions, including multimodal interfaces, on device LLMs, and improved orchestration for democratizing app creation with AI. Our findings indicate that while zero code LLM platforms greatly reduce the barrier to creating AI powered applications, they still face challenges in flexibility and reliability. Overall, the landscape is rapidly evolving, offering exciting opportunities to empower non programmers to create sophisticated software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v5">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. Our code is available at https://github.com/IANNXANG/RuscaRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11791v2">SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Rigorous security-focused evaluation of large language model (LLM) agents is imperative for establishing trust in their safe deployment throughout the software development lifecycle. However, existing benchmarks largely rely on synthetic challenges or simplified vulnerability datasets that fail to capture the complexity and ambiguity encountered by security engineers in practice. We introduce SEC-bench, the first fully automated benchmarking framework for evaluating LLM agents on authentic security engineering tasks. SEC-bench employs a novel multi-agent scaffold that automatically constructs code repositories with harnesses, reproduces vulnerabilities in isolated environments, and generates gold patches for reliable evaluation. Our framework automatically creates high-quality software vulnerability datasets with reproducible artifacts at a cost of only $0.87 per instance. Using SEC-bench, we implement two critical software security tasks to rigorously evaluate LLM agents' capabilities: proof-of-concept (PoC) generation and vulnerability patching. A comprehensive evaluation of state-of-the-art LLM code agents reveals significant performance gaps, achieving at most 18.0% success in PoC generation and 34.0% in vulnerability patching on our complete dataset. These results highlight the crucial steps needed toward developing LLM agents that are more practical, intelligent, and autonomous for security engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19733v1">Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factorized hypernetwork framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to 26x fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18129v2">GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Github with code and benchmark set: https://github.com/Solirinai/GeoBenchX
    </div>
    <details class="paper-abstract">
      This paper establishes a benchmark for evaluating tool-calling capabilities of large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess eight commercial LLMs (Claude Sonnet 3.5 and 4, Claude Haiku 3.5, Gemini 2.0 Flash, Gemini 2.5 Pro Preview, GPT-4o, GPT-4.1 and o4-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks in four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test rejection accuracy. We develop a LLM-as-Judge evaluation framework to compare agent solutions against reference solutions. Results show o4-mini and Claude 3.5 Sonnet achieve the best overall performance, OpenAI's GPT-4.1, GPT-4o and Google's Gemini 2.5 Pro Preview do not fall far behind, but the last two are more efficient in identifying unsolvable tasks. Claude Sonnet 4, due its preference to provide any solution rather than reject a task, proved to be less accurate. We observe significant differences in token usage, with Anthropic models consuming more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources (available at https://github.com/Solirinai/GeoBenchX), providing one more standardized method for the ongoing evaluation of LLMs for GeoAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19676v1">CircuitGuard: Mitigating LLM Memorization in RTL Code Generation Against IP Leakage</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in generative tasks, including register-transfer level (RTL) hardware synthesis. However, their tendency to memorize training data poses critical risks when proprietary or security-sensitive designs are unintentionally exposed during inference. While prior work has examined memorization in natural language, RTL introduces unique challenges: In RTL, structurally different implementations (e.g., behavioral vs. gate-level descriptions) can realize the same hardware, leading to intellectual property (IP) leakage (full or partial) even without verbatim overlap. Conversely, even small syntactic variations (e.g., operator precedence or blocking vs. non-blocking assignments) can drastically alter circuit behavior, making correctness preservation especially challenging. In this work, we systematically study memorization in RTL code generation and propose CircuitGuard, a defense strategy that balances leakage reduction with correctness preservation. CircuitGuard (1) introduces a novel RTL-aware similarity metric that captures both structural and functional equivalence beyond surface-level overlap, and (2) develops an activation-level steering method that identifies and attenuates transformer components most responsible for memorization. Our empirical evaluation demonstrates that CircuitGuard identifies (and isolates) 275 memorization-critical features across layers 18-28 of Llama 3.1-8B model, achieving up to 80% reduction in semantic similarity to proprietary patterns while maintaining generation quality. CircuitGuard further shows 78-85% cross-domain transfer effectiveness, enabling robust memorization mitigation across circuit categories without retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19669v1">DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19661v1">AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 13 pages, 10 pages
    </div>
    <details class="paper-abstract">
      Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15268v2">IM-Chat: A Multi-agent LLM Framework Integrating Tool-Calling and Diffusion Modeling for Knowledge Transfer in Injection Molding Industry</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The injection molding industry faces critical challenges in preserving and transferring field knowledge, particularly as experienced workers retire and multilingual barriers hinder effective communication. This study introduces IM-Chat, a multi-agent framework based on large language models (LLMs), designed to facilitate knowledge transfer in injection molding. IM-Chat integrates both limited documented knowledge (e.g., troubleshooting tables, manuals) and extensive field data modeled through a data-driven process condition generator that infers optimal manufacturing settings from environmental inputs such as temperature and humidity, enabling robust and context-aware task resolution. By adopting a retrieval-augmented generation (RAG) strategy and tool-calling agents within a modular architecture, IM-Chat ensures adaptability without the need for fine-tuning. Performance was assessed across 100 single-tool and 60 hybrid tasks for GPT-4o, GPT-4o-mini, and GPT-3.5-turbo by domain experts using a 10-point rubric focused on relevance and correctness, and was further supplemented by automated evaluation using GPT-4o guided by a domain-adapted instruction prompt. The evaluation results indicate that more capable models tend to achieve higher accuracy, particularly in complex, tool-integrated scenarios. In addition, compared with the fine-tuned single-agent LLM, IM-Chat demonstrated superior accuracy, particularly in quantitative reasoning, and greater scalability in handling multiple information sources. Overall, these findings demonstrate the viability of multi-agent LLM systems for industrial knowledge workflows and establish IM-Chat as a scalable and generalizable approach to AI-assisted decision support in manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16551v3">From Reviews to Actionable Insights: An LLM-Based Approach for Attribute and Feature Extraction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      This research proposes a systematic, large language model (LLM) approach for extracting product and service attributes, features, and associated sentiments from customer reviews. Grounded in marketing theory, the framework distinguishes perceptual attributes from actionable features, producing interpretable and managerially actionable insights. We apply the methodology to 20,000 Yelp reviews of Starbucks stores and evaluate eight prompt variants on a random subset of reviews. Model performance is assessed through agreement with human annotations and predictive validity for customer ratings. Results show high consistency between LLMs and human coders and strong predictive validity, confirming the reliability of the approach. Human coders required a median of six minutes per review, whereas the LLM processed each in two seconds, delivering comparable insights at a scale unattainable through manual coding. Managerially, the analysis identifies attributes and features that most strongly influence customer satisfaction and their associated sentiments, enabling firms to pinpoint "joy points," address "pain points," and design targeted interventions. We demonstrate how structured review data can power an actionable marketing dashboard that tracks sentiment over time and across stores, benchmarks performance, and highlights high-leverage features for improvement. Simulations indicate that enhancing sentiment for key service features could yield 1-2% average revenue gains per store.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19507v1">Teaming LLMs to Detect and Mitigate Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted to NeurIPS 2025 workshop on Reliable ML from Unreliable Data
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated state-of-the-art results in large language model (LLM) hallucination detection and mitigation through consistency-based approaches which involve aggregating multiple responses sampled from a single LLM for a given prompt. These approaches help offset limitations stemming from the imperfect data on which LLMs are trained, which includes biases and under-representation of information required at deployment time among other limitations which can lead to hallucinations. We show that extending these single-model consistency methods to combine responses from multiple LLMs with different training data, training schemes and model architectures can result in substantial further improvements in hallucination detection and mitigation capabilities beyond their single-model consistency counterparts. We evaluate this \emph{consortium consistency} approach across many model teams from a pool of 15 LLMs and explore under what conditions it is beneficial to team together different LLMs in this manner. Further, we show that these performance improvements often come with reduced inference costs, offsetting a significant drawback with single-model consistency methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v5">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted by EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19462v1">AegisMCP: Online Graph Intrusion Detection for Tool-Augmented LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      In this work, we study security of Model Context Protocol (MCP) agent toolchains and their applications in smart homes. We introduce AegisMCP, a protocol-level intrusion detector. Our contributions are: (i) a minimal attack suite spanning instruction-driven escalation, chain-of-tool exfiltration, malicious MCP server registration, and persistence; (ii) NEBULA-Schema (Network-Edge Behavioral Learning for Untrusted LLM Agents), a reusable protocol-level instrumentation that represents MCP activity as a streaming heterogeneous temporal graph over agents, MCP servers, tools, devices, remotes, and sessions; and (iii) a CPU-only streaming detector that fuses novelty, session-DAG structure, and attribute cues for near-real-time edge inference, with optional fusion of local prompt-guardrail signals. On an emulated smart-home testbed spanning multiple MCP stacks and a physical bench, AegisMCP achieves sub-second per-window model inference and end-to-end alerting. The latency of AegisMCP is consistently sub-second on Intel N150-class edge hardware, while outperforming traffic-only and sequence baselines; ablations confirm the importance of DAG and install/permission signals. We release code, schemas, and generators for reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19438v1">AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Autonomous Driving Systems (ADS) are safety-critical, where failures can be severe. While Metamorphic Testing (MT) is effective for fault detection in ADS, existing methods rely heavily on manual effort and lack automation. We present AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that automates the extraction of Metamorphic Relations (MRs) from local traffic rules and the generation of valid follow-up test cases. AutoMT leverages LLMs to extract MRs from traffic rules in Gherkin syntax using a predefined ontology. A vision-language agent analyzes scenarios, and a search agent retrieves suitable MRs from a RAG-based database to generate follow-up cases via computer vision. Experiments show that AutoMT achieves up to 5 x higher test diversity in follow-up case generation compared to the best baseline (manual expert-defined MRs) in terms of validation rate, and detects up to 20.55% more behavioral violations. While manual MT relies on a fixed set of predefined rules, AutoMT automatically extracts diverse metamorphic relations that augment real-world datasets and help uncover corner cases often missed during in-field testing and data collection. Its modular architecture separating MR extraction, filtering, and test generation supports integration into industrial pipelines and potentially enables simulation-based testing to systematically cover underrepresented or safety-critical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19420v1">Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16062v2">Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 47 pages, 25 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: https://correctbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01349v4">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19361v1">AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.
    </details>
</div>
