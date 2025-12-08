# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02261v1">TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      LLM-based trading agents are increasingly deployed in real-world financial markets to perform autonomous analysis and execution. However, their reliability and robustness under adversarial or faulty conditions remain largely unexamined, despite operating in high-risk, irreversible financial environments. We propose TradeTrap, a unified evaluation framework for systematically stress-testing both adaptive and procedural autonomous trading agents. TradeTrap targets four core components of autonomous trading agents: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, and evaluates their robustness under controlled system-level perturbations. All evaluations are conducted in a closed-loop historical backtesting setting on real US equity market data with identical initial conditions, enabling fair and reproducible comparisons across agents and attacks. Extensive experiments show that small perturbations at a single component can propagate through the agent decision loop and induce extreme concentration, runaway exposure, and large portfolio drawdowns across both agent types, demonstrating that current autonomous trading agents can be systematically misled at the system level. Our code is available at https://github.com/Yanlewen/TradeTrap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17117v6">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact conceptual categories that balance compression with semantic richness. Large Language Models (LLMs) exhibit impressive linguistic abilities, but whether they navigate this same compression-meaning trade-off remains unclear. We apply an Information Bottleneck framework to compare human conceptual structure with embeddings from 40+ LLMs using classic categorization benchmarks. We find that LLMs broadly align with human category boundaries, yet fall short on fine-grained semantic distinctions. Unlike humans, who maintain ``inefficient'' representations that preserve contextual nuance, LLMs aggressively compress, achieving more optimal information-theoretic compression at the cost of semantic richness. Surprisingly, encoder models outperform much larger decoder models in human alignment, suggesting that understanding and generation rely on distinct representational mechanisms. Training-dynamics analysis reveals a two-phase trajectory: rapid initial concept formation followed by architectural reorganization, during which semantic processing migrates from deep to mid-network layers as the model discovers increasingly efficient, sparser encodings. These divergent strategies, where LLMs optimize for compression and humans for adaptive utility, reveal fundamental differences between artificial and natural intelligence. This highlights the need for models that preserve the conceptual ``inefficiencies'' essential for human-like understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02230v1">Benchmarking LLM Agents for Wealth-Management Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 56 pages, 8 figures, The University of Edinburgh
    </div>
    <details class="paper-abstract">
      Modern work relies on an assortment of digital collaboration tools, yet routine processes continue to suffer from human error and delay. To address this gap, this dissertation extends TheAgentCompany with a finance-focused environment and investigates whether a general purpose LLM agent can complete representative wealth-management tasks both accurately and economically. This study introduces synthetic domain data, enriches colleague simulations, and prototypes an automatic task-generation pipeline. The study aims to create and assess an evaluation set that can meaningfully measure an agent's fitness for assistant-level wealth management work. We construct a benchmark of 12 task-pairs for wealth management assistants spanning retrieval, analysis, and synthesis/communication, with explicit acceptance criteria and deterministic graders. We seeded a set of new finance-specific data and introduced a high vs. low-autonomy variant of every task. The paper concluded that agents are limited less by mathematical reasoning and more so by end-to-end workflow reliability, and meaningfully affected by autonomy level, and that incorrect evaluation of models have hindered benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02228v1">STRIDE: A Systematic Framework for Selecting AI Modalities -- Agentic AI, AI Assistants, or LLM Calls</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 10 pages, 4 Figures, 5 Tables Paper presented at NeurIPS 2025 LAW workshop: Bridging Language, Agent, and World Models
    </div>
    <details class="paper-abstract">
      The rapid shift from stateless large language models (LLMs) to autonomous, goal-driven agents raises a central question: When is agentic AI truly necessary? While agents enable multi-step reasoning, persistent memory, and tool orchestration, deploying them indiscriminately leads to higher cost, complexity, and risk. We present STRIDE (Systematic Task Reasoning Intelligence Deployment Evaluator), a framework that provides principled recommendations for selecting between three modalities: (i) direct LLM calls, (ii) guided AI assistants, and (iii) fully autonomous agentic AI. STRIDE integrates structured task decomposition, dynamism attribution, and self-reflection requirement analysis to produce an Agentic Suitability Score, ensuring that full agentic autonomy is reserved for tasks with inherent dynamism or evolving context. Evaluated across 30 real-world tasks spanning SRE, compliance, and enterprise automation, STRIDE achieved 92% accuracy in modality selection, reduced unnecessary agent deployments by 45%, and cut resource costs by 37%. Expert validation over six months in SRE and compliance domains confirmed its practical utility, with domain specialists agreeing that STRIDE effectively distinguishes between tasks requiring simple LLM calls, guided assistants, or full agentic autonomy. This work reframes agent adoption as a necessity-driven design decision, ensuring autonomy is applied only when its benefits justify the costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08093v3">Evolution and compression in LLMs: On the emergence of human-aligned categorization</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Accepted at CogInterp: Interpreting Cognition in Deep Learning Models Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Converging evidence suggests that human systems of semantic categories achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy tradeoff. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-aligned semantic systems? To address this question, we focus on color categorization -- a key testbed of cognitive theories of categorization with uniquely rich human data -- and replicate with LLMs two influential human studies. First, we conduct an English color-naming study, showing that LLMs vary widely in their complexity and English-alignment, with larger instruction-tuned models achieving better alignment and IB-efficiency. Second, to test whether these LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via a method we refer to as Iterated in-Context Language Learning (IICLL). We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency. However, only a model with strongest in-context capabilities (Gemini 2.0) is able to recapitulate the wide range of near-optimal IB-tradeoffs observed in humans, while other state-of-the-art models converge to low-complexity solutions. These findings demonstrate how human-aligned semantic categories can emerge in LLMs via the same fundamental principle that underlies semantic efficiency in humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14233v2">Apertus: Democratizing Open and Compliant LLMs for Global Language Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting `robots.txt` exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03100v1">Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) and Supervised Finetuning (SFT) have become the predominant paradigms for equipping Large Language Models (LLMs) with external knowledge for diverse, knowledge-intensive tasks. However, while such knowledge injection improves performance, it also exposes new attack surfaces. Membership Inference Attacks (MIAs), which aim to determine whether a given data sample was included in a model's training set, pose serious threats to privacy and trust in sensitive domains. To this end, we first systematically evaluate the vulnerability of RAG- and SFT-based LLMs to various MIAs. Then, to address the privacy risk, we further introduce a novel, model-agnostic defense framework, Ensemble Privacy Defense (EPD), which aggregates and evaluates the outputs of a knowledge-injected LLM, a base LLM, and a dedicated judge model to enhance resistance against MIAs. Comprehensive experiments show that, on average, EPD reduces MIA success by up to 27.8\% for SFT and 526.3\% for RAG compared to inference-time baseline, while maintaining answer quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02002v1">LLM-Driven Corrective Robot Operation Code Generation with Static Text-Based Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large language models (LLMs) have demonstrated their promising capabilities of generating robot operation code to enable LLM-driven robots. To enhance the reliability of operation code generated by LLMs, corrective designs with feedback from the observation of executing code have been increasingly adopted in existing research. However, the code execution in these designs relies on either a physical experiment or a customized simulation environment, which limits their deployment due to the high configuration effort of the environment and the potential long execution time. In this paper, we explore the possibility of directly leveraging LLM to enable static simulation of robot operation code, and then leverage it to design a new reliable LLM-driven corrective robot operation code generation framework. Our framework configures the LLM as a static simulator with enhanced capabilities that reliably simulate robot code execution by interpreting actions, reasoning over state transitions, analyzing execution outcomes, and generating se- mantic observations that accurately capture trajectory dynamics. To validate the performance of our framework, we performed experiments on various operation tasks for different robots, including UAVs and small ground vehicles. The experiment results not only demonstrated the high accuracy of our static text-based simulation but also the reliable code generation of our LLM-driven corrective framework, which achieves a comparable performance with state-of-the-art research while does not rely on dynamic code execution using physical experiments or simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01992v1">LLM CHESS: Benchmarking Reasoning and Instruction-Following in LLMs through Chess</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We introduce LLM CHESS, an evaluation framework designed to probe the generalization of reasoning and instruction-following abilities in large language models (LLMs) through extended agentic interaction in the domain of chess. We rank over 50 open and closed source models by playing against a random opponent using a range of behavioral metrics, including win and loss rates, move quality, move legality, hallucinated actions, and game duration. For a subset of top reasoning models, we derive an Elo estimate by playing against a chess engine with variably configured skill, which allows for comparisons between models in an easily understandable way. Despite the simplicity of the instruction-following task and the weakness of the opponent, many state-of-the-art models struggle to complete games or achieve consistent wins. Similar to other benchmarks on complex reasoning tasks, our experiments reveal a clear separation between reasoning and non-reasoning models. However, unlike existing static benchmarks, the stochastic and dynamic nature of LLM CHESS uniquely reduces overfitting and memorization while preventing benchmark saturation, proving difficult even for top reasoning models. To support future work on evaluating reasoning and instruction-following in LLMs, we release our experimental framework, a public leaderboard, and a dataset of associated games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12286v4">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce two diagnostic tasks: file path identification from issue descriptions alone and ground truth function reproduction with only the current file context and issue description to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. Similar patterns are also observed for the function reproduction task, where the verbatim similarity is much higher on SWE-Bench Verified than on other similar coding benchmarks (up to 35% consecutive 5-gram accuracy on SWE-Bench Verified and Full, but only up to 18% for tasks in other benchmarks). These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01925v1">Rectifying LLM Thought from Lens of Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (Rectifying Process-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01909v1">Latent Debate: A Surrogate Framework for Interpreting LLM Thinking</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Understanding the internal thinking process of Large Language Models (LLMs) and the cause of hallucinations remains a key challenge. To this end, we introduce latent debate, a novel framework for interpreting model predictions through the lens of implicit internal arguments. Unlike the current work of self-consistency and multi-agent debate, which relies on explicit debates among multiple answers or multiple models, latent debate captures the hidden supporting and attacking signals that arise within a single model during a single inference. We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs on True/False prediction tasks. Empirical studies demonstrate that latent debate is a faithful structured surrogate model that has highly consistent predictions with the original LLM. Beyond interpretability, we demonstrate that latent debate provides a strong baseline for hallucination detection. Further analysis reveals strong correlations between hallucinations and debate patterns, such as a high degree of latent debates in the middle layers is linked to a higher risk of hallucinations. These findings position latent debate as a potential framework for understanding internal mechanisms of LLMs, especially for scenarios where internal (dis)agreements appear during the inference steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20075v4">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 21 pages, main paper 9 pages
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present Calgacus, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01830v1">OpenREAD: Reinforced Open-Ended Reasoing for End-to-End Autonomous Driving with LLM-as-Critic</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01786v1">Who Judges the Judge? LLM Jury-on-Demand: Building Trustworthy LLM Evaluation Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 66 pages, 22 figures, 37 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integrated into high-stakes domains, there is a growing need for evaluation methods that are both scalable for real-time deployment and reliable for critical decision-making. While human evaluation is reliable, it is slow and costly. Single LLM judges are biased, and static juries lack adaptability. To overcome these limitations, we propose LLM Jury-on-Demand - a dynamic, learning-based framework for scalable and context-aware evaluation. Our method trains a set of reliability predictors to assess when LLM judges will agree with human experts, leveraging token distributions, embeddings, and structural input features. This enables a fully adaptive evaluation where, for each data point, an optimal jury of the most reliable judges is dynamically selected, and their scores are aggregated using their reliability as weights. Experiments on summarization and RAG benchmarks show that our dynamic jury system achieves significantly higher correlation with human judgment than both single-judge and static-jury baselines. These results highlight the promise of adaptive, learning-based juries for building scalable, more reliable and trustworthy evaluation systems for modern LLMs in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13189v2">Multimodal "Puppeteer": Exploring Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 This work is under peer review
    </div>
    <details class="paper-abstract">
      The integration of robotics and augmented reality (AR) offers promising opportunities to enhance human-robot interaction (HRI) by making teleoperation more transparent, spatially grounded, and intuitive. We present a head-mounted AR "puppeteer" framework in which users control a physical robot via interacting with its virtual counterpart robot using large language model (LLM)-driven voice commands and hand-gesture interaction on the Meta Quest 3. In a within-subject user study with 42 participants performing an AR-based robotic pick-and-place pattern-matching task, we compare two interaction conditions: gesture-only (GO) and combined voice+gesture (VG). Our results show that GO currently provides more reliable and efficient control for this time-critical task, while VG introduces additional flexibility but also latency and recognition issues that can increase workload. We further explore how prior robotics experience shapes participants' perceptions of each modality. Based on these findings, we distill a set of evidence-based design guidelines for AR puppeteer metaphoric robot teleoperation, implicating multimodality as an adaptive strategy that must balance efficiency, robustness, and user expertise rather than assuming that additional modalities are universally beneficial. Our work contributes empirical insights into how multimodal (voice+gesture) interaction influences task efficiency, usability, and user experience in AR-based HRI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24616v4">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 short version
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01735v1">Automating modeling in mechanics: LLMs as designers of physics-constrained neural networks for constitutive modeling of materials</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Currently under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agentic frameworks increasingly adopt the paradigm of dynamically generating task-specific agents. We suggest that not only agents but also specialized software modules for scientific and engineering tasks can be generated on demand. We demonstrate this concept in the field of solid mechanics. There, so-called constitutive models are required to describe the relationship between mechanical stress and body deformation. Constitutive models are essential for both the scientific understanding and industrial application of materials. However, even recent data-driven methods of constitutive modeling, such as constitutive artificial neural networks (CANNs), still require substantial expert knowledge and human labor. We present a framework in which an LLM generates a CANN on demand, tailored to a given material class and dataset provided by the user. The framework covers LLM-based architecture selection, integration of physical constraints, and complete code generation. Evaluation on three benchmark problems demonstrates that LLM-generated CANNs achieve accuracy comparable to or greater than manually engineered counterparts, while also exhibiting reliable generalization to unseen loading scenarios and extrapolation to large deformations. These findings indicate that LLM-based generation of physics-constrained neural networks can substantially reduce the expertise required for constitutive modeling and represent a step toward practical end-to-end automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01693v1">LLM-Driven Multi-Agent Curation and Expansion of Metal-Organic Frameworks Database</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Metal-organic framework (MOF) databases have grown rapidly through experimental deposition and large-scale literature extraction, but recent analyses show that nearly half of their entries contain substantial structural errors. These inaccuracies propagate through high-throughput screening and machine-learning workflows, limiting the reliability of data-driven MOF discovery. Correcting such errors is exceptionally difficult because true repairs require integrating crystallographic files, synthesis descriptions, and contextual evidence scattered across the literature. Here we introduce LitMOF, a large language model-driven multi-agent framework that validates crystallographic information directly from the original literature and cross-validates it with database entries to repair structural errors. Applying LitMOF to the experimental MOF database (the CSD MOF Subset), we constructed LitMOF-DB, a curated set 118,464 computation-ready structures, including corrections of 69% (6,161 MOFs) of the invalid MOFs in the latest CoRE MOF database. Additionally, the system uncovered 12,646 experimentally reported MOFs absent from existing resources, substantially expanding the known experimental design space. This work establishes a scalable pathway toward self-correcting scientific databases and a generalizable paradigm for LLM-driven curation in materials science.
    </details>
</div>
