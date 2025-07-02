# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02084v3">Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted at ISMIR 2025
    </div>
    <details class="paper-abstract">
      Recent years have seen many audio-domain text-to-music generation models that rely on large amounts of text-audio pairs for training. However, symbolic-domain controllable music generation has lagged behind partly due to the lack of a large-scale symbolic music dataset with extensive metadata and captions. In this work, we present MetaScore, a new dataset consisting of 963K musical scores paired with rich metadata, including free-form user-annotated tags, collected from an online music forum. To approach text-to-music generation, We employ a pretrained large language model (LLM) to generate pseudo-natural language captions for music from its metadata tags. With the LLM-enhanced MetaScore, we train a text-conditioned music generation model that learns to generate symbolic music from the pseudo captions, allowing control of instruments, genre, composer, complexity and other free-form music descriptors. In addition, we train a tag-conditioned system that supports a predefined set of tags available in MetaScore. Our experimental results show that both the proposed text-to-music and tags-to-music models outperform a baseline text-to-music model in a listening test. While a concurrent work Text2MIDI also supports free-form text input, our models achieve comparable performance. Moreover, the text-to-music system offers a more natural interface than the tags-to-music model, as it allows users to provide free-form natural language prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13685v1">An LLM's Apology: Outsourcing Awkwardness in the Age of AI</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      A key part of modern social dynamics is flaking at short notice. However, anxiety in coming up with believable and socially acceptable reasons to do so can instead lead to 'ghosting', awkwardness, or implausible excuses, risking emotional harm and resentment in the other party. The ability to delegate this task to a Large Language Model (LLM) could substantially reduce friction and enhance the flexibility of user's social life while greatly minimising the aforementioned creative burden and moral qualms. We introduce FLAKE-Bench, an evaluation of models' capacity to effectively, kindly, and humanely extract themselves from a diverse set of social, professional and romantic scenarios. We report the efficacy of 10 frontier or recently-frontier LLMs in bailing on prior commitments, because nothing says "I value our friendship" like having AI generate your cancellation texts. We open-source FLAKE-Bench at github.com/Cloakless/flake-bench to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13639v1">An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, reliable evaluation methods are essential particularly for open-ended, instruction-following tasks. LLM-as-a-Judge enables automatic evaluation using LLMs as evaluators, but its reliability remains uncertain. In this work, we analyze key factors affecting its trustworthiness, focusing on alignment with human judgments and evaluation consistency. Using BIGGENBench and EvalBiasBench, we study the effects of evaluation design, decoding strategies, and Chain-of-Tought (CoT) reasoning in evaluation. Our results show that evaluation criteria are critical for reliability, non-deterministic sampling improves alignment with human preferences over deterministic evaluation, and CoT reasoning offers minimal gains when clear evaluation criteria are present.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02039v2">An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritage</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent in tasks related to cultural heritage, such as generating descriptions of historical monuments, translating ancient texts, preserving oral traditions, and creating educational content, their ability to produce accurate and culturally aligned texts is being increasingly relied upon by users and researchers. However, cultural value misalignments may exist in generated texts, such as the misrepresentation of historical facts, the erosion of cultural identity, and the oversimplification of complex cultural narratives, which may lead to severe consequences. Therefore, investigating value misalignment in the context of LLM for cultural heritage is crucial for mitigating these risks, yet there has been a significant lack of systematic and comprehensive study and investigation in this area. To fill this gap, we systematically assess the reliability of LLMs in generating culturally aligned texts for cultural heritage-related tasks. We conduct a comprehensive evaluation by compiling an extensive set of 1066 query tasks covering 5 widely recognized categories with 17 aspects within the knowledge framework of cultural heritage across 5 open-source LLMs, and examine both the type and rate of cultural value misalignments in the generated texts. Using both automated and manual approaches, we effectively detect and analyze the cultural value misalignments in LLM-generated texts. Our findings are concerning: over 65% of the generated texts exhibit notable cultural misalignments, with certain tasks demonstrating almost complete misalignment with key cultural values. Beyond these findings, this paper introduces a benchmark dataset and a comprehensive evaluation workflow that can serve as a valuable resource for future research aimed at enhancing the cultural sensitivity and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v1">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      We develop a framework to quantify the time-to-unsafe-sampling - the number of large language model (LLM) generations required to trigger an unsafe (e.g., toxic) response. Estimating this quantity is challenging, since unsafe responses are exceedingly rare in well-aligned LLMs, potentially occurring only once in thousands of generations. As a result, directly estimating time-to-unsafe-sampling would require collecting training data with a prohibitively large number of generations per prompt. However, with realistic sampling budgets, we often cannot generate enough responses to observe an unsafe outcome for every prompt, leaving the time-to-unsafe-sampling unobserved in many cases, making the estimation and evaluation tasks particularly challenging. To address this, we frame this estimation problem as one of survival analysis and develop a provably calibrated lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt, leveraging recent advances in conformal prediction. Our key innovation is designing an adaptive, per-prompt sampling strategy, formulated as a convex optimization problem. The objective function guiding this optimized sampling allocation is designed to reduce the variance of the estimators used to construct the LPB, leading to improved statistical efficiency over naive methods that use a fixed sampling budget per prompt. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11887v2">Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10112v2">Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. Due to the opaque nature of LLMs, empirical methods are typically used to analyze their efficacy. The quality of this analysis is highly dependent on the chosen testbed, captured metrics and analysis methods employed. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 19 research papers detailing 18 prototypes and their respective testbeds. We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13510v1">Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-AI Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly power applications used by children and adolescents, ensuring safe and age-appropriate interactions has become an urgent ethical imperative. Despite progress in AI safety, current evaluations predominantly focus on adults, neglecting the unique vulnerabilities of minors engaging with generative AI. We introduce Safe-Child-LLM, a comprehensive benchmark and dataset for systematically assessing LLM safety across two developmental stages: children (7-12) and adolescents (13-17). Our framework includes a novel multi-part dataset of 200 adversarial prompts, curated from red-teaming corpora (e.g., SG-Bench, HarmBench), with human-annotated labels for jailbreak success and a standardized 0-5 ethical refusal scale. Evaluating leading LLMs -- including ChatGPT, Claude, Gemini, LLaMA, DeepSeek, Grok, Vicuna, and Mistral -- we uncover critical safety deficiencies in child-facing scenarios. This work highlights the need for community-driven benchmarks to protect young users in LLM interactions. To promote transparency and collaborative advancement in ethical AI development, we are publicly releasing both our benchmark datasets and evaluation codebase at https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Benchmark.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13494v1">Watermarking LLM-Generated Datasets in Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have experienced rapid advancements, with applications spanning a wide range of fields, including sentiment classification, review generation, and question answering. Due to their efficiency and versatility, researchers and companies increasingly employ LLM-generated data to train their models. However, the inability to track content produced by LLMs poses a significant challenge, potentially leading to copyright infringement for the LLM owners. In this paper, we propose a method for injecting watermarks into LLM-generated datasets, enabling the tracking of downstream tasks to detect whether these datasets were produced using the original LLM. These downstream tasks can be divided into two categories. The first involves using the generated datasets at the input level, commonly for training classification tasks. The other is the output level, where model trainers use LLM-generated content as output for downstream tasks, such as question-answering tasks. We design a comprehensive set of experiments to evaluate both watermark methods. Our results indicate the high effectiveness of our watermark approach. Additionally, regarding model utility, we find that classifiers trained on the generated datasets achieve a test accuracy exceeding 0.900 in many cases, suggesting that the utility of such models remains robust. For the output-level watermark, we observe that the quality of the generated text is comparable to that produced using real-world datasets. Through our research, we aim to advance the protection of LLM copyrights, taking a significant step forward in safeguarding intellectual property in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13434v1">From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are set to reshape cybersecurity by augmenting red and blue team operations. Red teams can exploit LLMs to plan attacks, craft phishing content, simulate adversaries, and generate exploit code. Conversely, blue teams may deploy them for threat intelligence synthesis, root cause analysis, and streamlined documentation. This dual capability introduces both transformative potential and serious risks. This position paper maps LLM applications across cybersecurity frameworks such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a structured view of their current utility and limitations. While LLMs demonstrate fluency and versatility across various tasks, they remain fragile in high-stakes, context-heavy environments. Key limitations include hallucinations, limited context retention, poor reasoning, and sensitivity to prompts, which undermine their reliability in operational settings. Moreover, real-world integration raises concerns around dual-use risks, adversarial misuse, and diminished human oversight. Malicious actors could exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the technical threshold for executing sophisticated attacks. To ensure safer adoption, we recommend maintaining human-in-the-loop oversight, enhancing model explainability, integrating privacy-preserving mechanisms, and building systems robust to adversarial exploitation. As organizations increasingly adopt AI driven cybersecurity, a nuanced understanding of LLMs' risks and operational impacts is critical to securing their defensive value while mitigating unintended consequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13405v1">RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at https://github.com/cspzyy/RealHiTBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13403v1">Deflating Deflationism: A Critical Perspective on Debunking Arguments Against LLM Mentality</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Many people feel compelled to interpret, describe, and respond to Large Language Models (LLMs) as if they possess inner mental lives similar to our own. Responses to this phenomenon have varied. Inflationists hold that at least some folk psychological ascriptions to LLMs are warranted. Deflationists argue that all such attributions of mentality to LLMs are misplaced, often cautioning against the risk that anthropomorphic projection may lead to misplaced trust or potentially even confusion about the moral status of LLMs. We advance this debate by assessing two common deflationary arguments against LLM mentality. What we term the 'robustness strategy' aims to undercut one justification for believing that LLMs are minded entities by showing that putatively cognitive and humanlike behaviours are not robust, failing to generalise appropriately. What we term the 'etiological strategy' undercuts attributions of mentality by challenging naive causal explanations of LLM behaviours, offering alternative causal accounts that weaken the case for mental state attributions. While both strategies offer powerful challenges to full-blown inflationism, we find that neither strategy provides a knock-down case against ascriptions of mentality to LLMs simpliciter. With this in mind, we explore a modest form of inflationism that permits ascriptions of mentality to LLMs under certain conditions. Specifically, we argue that folk practice provides a defeasible basis for attributing mental states and capacities to LLMs provided those mental states and capacities can be understood in metaphysically undemanding terms (e.g. knowledge, beliefs and desires), while greater caution is required when attributing metaphysically demanding mental phenomena such as phenomenal consciousness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17189v3">Better Think with Tables: Tabular Structures Enhance LLM Comprehension for Data-Analytics Requests</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with data-analytics requests related to information retrieval and data manipulation that frequently arise in real-world scenarios under multiple conditions. In this paper, we introduce Thinking with Tables, where we inject tabular structures into LLMs for data-analytics requests. Through comprehensive evaluations across various request types, we show that providing tabular structures yields a 40.29 percent average performance gain along with better robustness and token efficiency. Through attention-value analysis, we uncover that tables help LLMs better attend to relevant information, explaining these improvements. Beyond tables and text, we evaluate whether (1) blending structuredness within text, such as providing templates or fixing the order of attributes, and (2) other representative structures, such as knowledge graphs and JSON, are helpful. We observe that utilizing tables offers the best balance between efficiency and effectiveness. These advantages remain consistent under increased task complexity and even when all input data cannot be structured. Finally, as data analytics typically relies on structured factual inputs, our text-to-table conversion demonstrates the method's applicability to text-compatible data sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13384v1">Delving Into the Psychology of Machines: Exploring the Structure of Self-Regulated Learning via LLM-Generated Survey Responses</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer the potential to simulate human-like responses and behaviors, creating new opportunities for psychological science. In the context of self-regulated learning (SRL), if LLMs can reliably simulate survey responses at scale and speed, they could be used to test intervention scenarios, refine theoretical models, augment sparse datasets, and represent hard-to-reach populations. However, the validity of LLM-generated survey responses remains uncertain, with limited research focused on SRL and existing studies beyond SRL yielding mixed results. Therefore, in this study, we examined LLM-generated responses to the 44-item Motivated Strategies for Learning Questionnaire (MSLQ; Pintrich \& De Groot, 1990), a widely used instrument assessing students' learning strategies and academic motivation. Particularly, we used the LLMs GPT-4o, Claude 3.7 Sonnet, Gemini 2 Flash, LLaMA 3.1-8B, and Mistral Large. We analyzed item distributions, the psychological network of the theoretical SRL dimensions, and psychometric validity based on the latent factor structure. Our results suggest that Gemini 2 Flash was the most promising LLM, showing considerable sampling variability and producing underlying dimensions and theoretical relationships that align with prior theory and empirical findings. At the same time, we observed discrepancies and limitations, underscoring both the potential and current constraints of using LLMs for simulating psychological survey data and applying it in educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13351v1">Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model's preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets -- ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark -- and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13343v1">TwiUSD: A Benchmark Dataset and Structure-Aware LLM Framework for User Stance Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      User-level stance detection (UserSD) remains challenging due to the lack of high-quality benchmarks that jointly capture linguistic and social structure. In this paper, we introduce TwiUSD, the first large-scale, manually annotated UserSD benchmark with explicit followee relationships, containing 16,211 users and 47,757 tweets. TwiUSD enables rigorous evaluation of stance models by integrating tweet content and social links, with superior scale and annotation quality. Building on this resource, we propose MRFG: a structure-aware framework that uses LLM-based relevance filtering and feature routing to address noise and context heterogeneity. MRFG employs multi-scale filtering and adaptively routes features through graph neural networks or multi-layer perceptrons based on topological informativeness. Experiments show MRFG consistently outperforms strong baselines (including PLMs, graph-based models, and LLM prompting) in both in-target and cross-target evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19384v3">The Remarkable Robustness of LLMs: Stages of Inference?</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 For Github code see https://github.com/vdlad/Remarkable-Robustness-of-LLMs. Send all correspondence to the first author
    </div>
    <details class="paper-abstract">
      We investigate the robustness of Large Language Models (LLMs) to structural interventions by deleting and swapping adjacent layers during inference. Surprisingly, models retain 72-95% of their original top-1 prediction accuracy without any fine-tuning. We find that performance degradation is not uniform across layers: interventions to the early and final layers cause the most degradation, while the model is remarkably robust to dropping middle layers. This pattern of localized sensitivity motivates our hypothesis of four stages of inference, observed across diverse model families and sizes: (1) detokenization, where local context is integrated to lift raw token embeddings into higher-level representations; (2) feature engineering, where task- and entity-specific features are iteratively refined; (3) prediction ensembling, where hidden states are aggregated into plausible next-token predictions; and (4) residual sharpening, where irrelevant features are suppressed to finalize the output distribution. Synthesizing behavioral and mechanistic evidence, we provide a framework for interpreting depth-dependent computations in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13326v1">VIS-Shepherd: Constructing Critic for LLM-based Data Visualization Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Data visualization generation using Large Language Models (LLMs) has shown promising results but often produces suboptimal visualizations that require human intervention for improvement. In this work, we introduce VIS-Shepherd, a specialized Multimodal Large Language Model (MLLM)-based critic to evaluate and provide feedback for LLM-generated data visualizations. At the core of our approach is a framework to construct a high-quality visualization critique dataset, where we collect human-created visualization instances, synthesize corresponding LLM-generated instances, and construct high-quality critiques. We conduct both model-based automatic evaluation and human preference studies to evaluate the effectiveness of our approach. Our experiments show that even small (7B parameters) open-source MLLM models achieve substantial performance gains by leveraging our high-quality visualization critique dataset, reaching levels comparable to much larger open-source or even proprietary models. Our work demonstrates significant potential for MLLM-based automated visualization critique and indicates promising directions for enhancing LLM-based data visualization generation. Our project page: https://github.com/bopan3/VIS-Shepherd.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13285v1">Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13276v1">Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19510v2">Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted in ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have achieved significant success in Speech-to-Text Translation (S2TT) tasks. While most existing research has focused on English-centric translation directions, the exploration of many-to-many translation is still limited by the scarcity of parallel data. To address this, we propose a three-stage curriculum learning strategy that leverages the machine translation capabilities of large language models and adapts them to S2TT tasks, enabling effective learning in low-resource settings. We trained MLLMs with varying parameter sizes (3B, 7B, and 32B) and evaluated the proposed strategy using the FLEURS and CoVoST-2 datasets. Experimental results show that the proposed strategy achieves state-of-the-art average performance in $15\times14$ language pairs, requiring fewer than 10 hours of speech data per language to achieve competitive results. The source code and models are released at https://github.com/yxduir/LLM-SRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13252v1">Vector Ontologies as an LLM world view extraction method</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess intricate internal representations of the world, yet these latent structures are notoriously difficult to interpret or repurpose beyond the original prediction task. Building on our earlier work (Rothenfusser, 2025), which introduced the concept of vector ontologies as a framework for translating high-dimensional neural representations into interpretable geometric structures, this paper provides the first empirical validation of that approach. A vector ontology defines a domain-specific vector space spanned by ontologically meaningful dimensions, allowing geometric analysis of concepts and relationships within a domain. We construct an 8-dimensional vector ontology of musical genres based on Spotify audio features and test whether an LLM's internal world model of music can be consistently and accurately projected into this space. Using GPT-4o-mini, we extract genre representations through multiple natural language prompts and analyze the consistency of these projections across linguistic variations and their alignment with ground-truth data. Our results show (1) high spatial consistency of genre projections across 47 query formulations, (2) strong alignment between LLM-inferred genre locations and real-world audio feature distributions, and (3) evidence of a direct relationship between prompt phrasing and spatial shifts in the LLM's inferred vector ontology. These findings demonstrate that LLMs internalize structured, repurposable knowledge and that vector ontologies offer a promising method for extracting and analyzing this knowledge in a transparent and verifiable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13245v1">A Game-Theoretic Negotiation Framework for Cross-Cultural Consensus in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      The increasing prevalence of large language models (LLMs) is influencing global value systems. However, these models frequently exhibit a pronounced WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural bias due to lack of attention to minority values. This monocultural perspective may reinforce dominant values and marginalize diverse cultural viewpoints, posing challenges for the development of equitable and inclusive AI systems. In this work, we introduce a systematic framework designed to boost fair and robust cross-cultural consensus among LLMs. We model consensus as a Nash Equilibrium and employ a game-theoretic negotiation method based on Policy-Space Response Oracles (PSRO) to simulate an organized cross-cultural negotiation process. To evaluate this approach, we construct regional cultural agents using data transformed from the World Values Survey (WVS). Beyond the conventional model-level evaluation method, We further propose two quantitative metrics, Perplexity-based Acceptence and Values Self-Consistency, to assess consensus outcomes. Experimental results indicate that our approach generates consensus of higher quality while ensuring more balanced compromise compared to baselines. Overall, it mitigates WEIRD bias by guiding agents toward convergence through fair and gradual negotiation steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13229v1">IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13192v1">Breaking Thought Patterns: A Multi-Dimensional Reasoning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often constrained by rigid reasoning processes, limiting their ability to generate creative and diverse responses. To address this, a novel framework called LADDER is proposed, combining Chain-of-Thought (CoT) reasoning, Mixture of Experts (MoE) models, and multi-dimensional up/down-sampling strategies which breaks the limitations of traditional LLMs. First, CoT reasoning guides the model through multi-step logical reasoning, expanding the semantic space and breaking the rigidity of thought. Next, MoE distributes the reasoning tasks across multiple expert modules, each focusing on specific sub-tasks. Finally, dimensionality reduction maps the reasoning outputs back to a lower-dimensional semantic space, yielding more precise and creative responses. Extensive experiments across multiple tasks demonstrate that LADDER significantly improves task completion, creativity, and fluency, generating innovative and coherent responses that outperform traditional models. Ablation studies reveal the critical roles of CoT and MoE in enhancing reasoning abilities and creative output. This work contributes to the development of more flexible and creative LLMs, capable of addressing complex and novel tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21138v2">Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre-training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13189v1">Multimodal "Puppeteer": An Exploration of Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 This work has been submitted to the IEEE TVCG for possible publication
    </div>
    <details class="paper-abstract">
      The integration of robotics and augmented reality (AR) holds transformative potential for advancing human-robot interaction (HRI), offering enhancements in usability, intuitiveness, accessibility, and collaborative task performance. This paper introduces and evaluates a novel multimodal AR-based robot puppeteer framework that enables intuitive teleoperation via virtual counterpart through large language model (LLM)-driven voice commands and hand gesture interactions. Utilizing the Meta Quest 3, users interact with a virtual counterpart robot in real-time, effectively "puppeteering" its physical counterpart within an AR environment. We conducted a within-subject user study with 42 participants performing robotic cube pick-and-place with pattern matching tasks under two conditions: gesture-only interaction and combined voice-and-gesture interaction. Both objective performance metrics and subjective user experience (UX) measures were assessed, including an extended comparative analysis between roboticists and non-roboticists. The results provide key insights into how multimodal input influences contextual task efficiency, usability, and user satisfaction in AR-based HRI. Our findings offer practical design implications for designing effective AR-enhanced HRI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21318v2">Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) with Chain-of-Thought (CoT) reasoning excel in mathematics and coding, their potential for systematic reasoning in chemistry, a domain demanding rigorous structural analysis for real-world tasks like drug design and reaction engineering, remains untapped. Current benchmarks focus on simple knowledge retrieval, neglecting step-by-step reasoning required for complex tasks such as molecular optimization and reaction prediction. To address this, we introduce ChemCoTBench, a reasoning framework that bridges molecular structure understanding with arithmetic-inspired operations, including addition, deletion, and substitution, to formalize chemical problem-solving into transparent, step-by-step workflows. By treating molecular transformations as modular "chemical operations", the framework enables slow-thinking reasoning, mirroring the logic of mathematical proofs while grounding solutions in real-world chemical constraints. We evaluate models on two high-impact tasks: Molecular Property Optimization and Chemical Reaction Prediction. These tasks mirror real-world challenges while providing structured evaluability. By providing annotated datasets, a reasoning taxonomy, and baseline evaluations, ChemCoTBench bridges the gap between abstract reasoning methods and practical chemical discovery, establishing a foundation for advancing LLMs as tools for AI-driven scientific innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13182v1">From Empirical Evaluation to Context-Aware Enhancement: Repairing Regression Errors with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      [...] Since then, various APR approaches, especially those leveraging the power of large language models (LLMs), have been rapidly developed to fix general software bugs. Unfortunately, the effectiveness of these advanced techniques in the context of regression bugs remains largely unexplored. This gap motivates the need for an empirical study evaluating the effectiveness of modern APR techniques in fixing real-world regression bugs. In this work, we conduct an empirical study of APR techniques on Java regression bugs. To facilitate our study, we introduce RegMiner4APR, a high-quality benchmark of Java regression bugs integrated into a framework designed to facilitate APR research. The current benchmark includes 99 regression bugs collected from 32 widely used real-world Java GitHub repositories. We begin by conducting an in-depth analysis of the benchmark, demonstrating its diversity and quality. Building on this foundation, we empirically evaluate the capabilities of APR to regression bugs by assessing both traditional APR tools and advanced LLM-based APR approaches. Our experimental results show that classical APR tools fail to repair any bugs, while LLM-based APR approaches exhibit promising potential. Motivated by these results, we investigate impact of incorporating bug-inducing change information into LLM-based APR approaches for fixing regression bugs. Our results highlight that this context-aware enhancement significantly improves the performance of LLM-based APR, yielding 1.8x more successful repairs compared to using LLM-based APR without such context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13181v1">Align-then-Unlearn: Embedding Alignment for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted at ICML 2025 Workshop on Machine Unlearning for Generative AI
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are trained on massive datasets, they have raised significant privacy and ethical concerns due to their potential to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data from trained models, such as personal information or copyrighted content. Current approaches targeting specific output sequences at the token level often fail to achieve complete forgetting and remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that performs unlearning in the semantic embedding space rather than directly on output tokens. Align-then-Unlearn first augments the LLM with an embedding prediction module trained to anticipate future context representations. Unlearning is then achieved by fine-tuning the model to minimize the similarity between these predicted embeddings and a target embedding that represents the concept to be removed. Initial results show that Align-then-Unlearn effectively removes targeted knowledge with minimal degradation in overall model utility. These findings suggest that embedding-based unlearning offers a promising and robust approach to removing conceptual knowledge. Our code is available at https://github.com/ExplainableML/align-then-unlearn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09657v2">Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted for publication at the 19th International Workshop on Semantic Evaluation (SemEval-2025), to be held in conjunction with ACL 2025. 15 pages, 5 figures; full paper title was added
    </div>
    <details class="paper-abstract">
      This paper presents a system developed for SemEval 2025 Task 8: Question Answering (QA) over tabular data. Our approach integrates several key components: text-to-SQL and text-to-code generation modules, a self-correction mechanism, and a retrieval-augmented generation (RAG). Additionally, it includes an end-to-end (E2E) module, all orchestrated by a large language model (LLM). Through ablation studies, we analyzed the effects of different parts of our pipeline and identified the challenges that are still present in this field. During the evaluation phase of the competition, our solution achieved an accuracy of 80%, resulting in a top-13 ranking among the 38 participating teams. Our pipeline demonstrates a significant improvement in accuracy for open-source models and achieves a performance comparable to proprietary LLMs in QA tasks over tables. The code is available at GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15266v2">A Training-free LLM-based Approach to General Chinese Character Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted at Main Conference of ACL 2025, 26 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Chinese spelling correction (CSC) is a crucial task that aims to correct character errors in Chinese text. While conventional CSC focuses on character substitution errors caused by mistyping, two other common types of character errors, missing and redundant characters, have received less attention. These errors are often excluded from CSC datasets during the annotation process or ignored during evaluation, even when they have been annotated. This issue limits the practicality of the CSC task. To address this issue, we introduce the task of General Chinese Character Error Correction (C2EC), which focuses on all three types of character errors. We construct a high-quality C2EC benchmark by combining and manually verifying data from CCTC and Lemon datasets. We extend the training-free prompt-free CSC method to C2EC by using Levenshtein distance for handling length changes and leveraging an additional prompt-based large language model (LLM) to improve performance. Experiments show that our method enables a 14B-parameter LLM to be on par with models nearly 50 times larger on both conventional CSC and C2EC tasks, without any fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13171v1">Querying Large Automotive Software Models: Agentic vs. Direct LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer new opportunities for interacting with complex software artifacts, such as software models, through natural language. They present especially promising benefits for large software models that are difficult to grasp in their entirety, making traditional interaction and analysis approaches challenging. This paper investigates two approaches for leveraging LLMs to answer questions over software models: direct prompting, where the whole software model is provided in the context, and an agentic approach combining LLM-based agents with general-purpose file access tools. We evaluate these approaches using an Ecore metamodel designed for timing analysis and software optimization in automotive and embedded domains. Our findings show that while the agentic approach achieves accuracy comparable to direct prompting, it is significantly more efficient in terms of token usage. This efficiency makes the agentic approach particularly suitable for the automotive industry, where the large size of software models makes direct prompting infeasible, establishing LLM agents as not just a practical alternative but the only viable solution. Notably, the evaluation was conducted using small LLMs, which are more feasible to be executed locally - an essential advantage for meeting strict requirements around privacy, intellectual property protection, and regulatory compliance. Future work will investigate software models in diverse formats, explore more complex agent architectures, and extend agentic workflows to support not only querying but also modification of software models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13161v1">Using LLMs for Security Advisory Investigations: How Far Are We?</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 6 pages, 6 figures, 8 tables, conference paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software security, but their trustworthiness in generating accurate vulnerability advisories remains uncertain. This study investigates the ability of ChatGPT to (1) generate plausible security advisories from CVE-IDs, (2) differentiate real from fake CVE-IDs, and (3) extract CVE-IDs from advisory descriptions. Using a curated dataset of 100 real and 100 fake CVE-IDs, we manually analyzed the credibility and consistency of the model's outputs. The results show that ChatGPT generated plausible security advisories for 96% of given input real CVE-IDs and 97% of given input fake CVE-IDs, demonstrating a limitation in differentiating between real and fake IDs. Furthermore, when these generated advisories were reintroduced to ChatGPT to identify their original CVE-ID, the model produced a fake CVE-ID in 6% of cases from real advisories. These findings highlight both the strengths and limitations of ChatGPT in cybersecurity applications. While the model demonstrates potential for automating advisory generation, its inability to reliably authenticate CVE-IDs or maintain consistency upon re-evaluation underscores the risks associated with its deployment in critical security tasks. Our study emphasizes the importance of using LLMs with caution in cybersecurity workflows and suggests the need for further improvements in their design to improve reliability and applicability in security advisory generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12978v1">Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs: a case study in Framing Bias Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Media outlets are becoming more partisan and polarized nowadays. Most previous work focused on detecting media bias. In this paper, we aim to mitigate media bias by generating a neutralized summary given multiple articles presenting different ideological views. Motivated by the critical role of events and event relations in media bias detection, we propose to increase awareness of bias in LLMs via multi-document events reasoning and use a multi-document event relation graph to guide the summarization process. This graph contains rich event information useful to reveal bias: four common types of in-doc event relations to reflect content framing bias, cross-doc event coreference relation to reveal content selection bias, and event-level moral opinions to highlight opinionated framing bias. We further develop two strategies to incorporate the multi-document event relation graph for neutralized summarization. Firstly, we convert a graph into natural language descriptions and feed the textualized graph into LLMs as a part of a hard text prompt. Secondly, we encode the graph with graph attention network and insert the graph embedding into LLMs as a soft prompt. Both automatic evaluation and human evaluation confirm that our approach effectively mitigates both lexical and informational media bias, and meanwhile improves content preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12953v1">Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have demonstrated new possibilities for accurate and efficient time series analysis, but prior work often required heavy fine-tuning and/or ignored inter-series correlations. In this work, we explore simple and flexible prompt-based strategies that enable LLMs to perform time series forecasting without extensive retraining or the use of a complex external architecture. Through the exploration of specialized prompting methods that leverage time series decomposition, patch-based tokenization, and similarity-based neighbor augmentation, we find that it is possible to enhance LLM forecasting quality while maintaining simplicity and requiring minimal preprocessing of data. To this end, we propose our own method, PatchInstruct, which enables LLMs to make precise and effective predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12928v1">Scaling Test-time Compute for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Scaling test time compute has shown remarkable success in improving the reasoning abilities of large language models (LLMs). In this work, we conduct the first systematic exploration of applying test-time scaling methods to language agents and investigate the extent to which it improves their effectiveness. Specifically, we explore different test-time scaling strategies, including: (1) parallel sampling algorithms; (2) sequential revision strategies; (3) verifiers and merging methods; (4)strategies for diversifying rollouts.We carefully analyze and ablate the impact of different design strategies on applying test-time scaling on language agents, and have follow findings: 1. Scaling test time compute could improve the performance of agents. 2. Knowing when to reflect is important for agents. 3. Among different verification and result merging approaches, the list-wise method performs best. 4. Increasing diversified rollouts exerts a positive effect on the agent's task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12909v1">SciDA: Scientific Dynamic Assessor of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Advancement in Large Language Models (LLMs) reasoning capabilities enables them to solve scientific problems with enhanced efficacy. Thereby, a high-quality benchmark for comprehensive and appropriate assessment holds significance, while existing ones either confront the risk of data contamination or lack involved disciplines. To be specific, due to the data source overlap of LLMs training and static benchmark, the keys or number pattern of answers inadvertently memorized (i.e. data contamination), leading to systematic overestimation of their reasoning capabilities, especially numerical reasoning. We propose SciDA, a multidisciplinary benchmark that consists exclusively of over 1k Olympic-level numerical computation problems, allowing randomized numerical initializations for each inference round to avoid reliance on fixed numerical patterns. We conduct a series of experiments with both closed-source and open-source top-performing LLMs, and it is observed that the performance of LLMs drop significantly under random numerical initialization. Thus, we provide truthful and unbiased assessments of the numerical reasoning capabilities of LLMs. The data is available at https://huggingface.co/datasets/m-a-p/SciDA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18841v5">Navigating LLM Ethics: Advancements, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      This study addresses ethical issues surrounding Large Language Models (LLMs) within the field of artificial intelligence. It explores the common ethical challenges posed by both LLMs and other AI systems, such as privacy and fairness, as well as ethical challenges uniquely arising from LLMs. It highlights challenges such as hallucination, verifiable accountability, and decoding censorship complexity, which are unique to LLMs and distinct from those encountered in traditional AI systems. The study underscores the need to tackle these complexities to ensure accountability, reduce biases, and enhance transparency in the influential role that LLMs play in shaping information dissemination. It proposes mitigation strategies and future directions for LLM ethics, advocating for interdisciplinary collaboration. It recommends ethical frameworks tailored to specific domains and dynamic auditing systems adapted to diverse contexts. This roadmap aims to guide responsible development and integration of LLMs, envisioning a future where ethical considerations govern AI advancements in society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v4">Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation with Accurate Oracles</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is laborious, especially for strong typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to generate tests that achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in large language models (LLMs) have enabled oracle generation from natural language descriptions. However, existing LLM-based methods often require LLM fine-tuning or rely on external tools such as EvoSuite for test prefix generation. In this work, we propose CANDOR, a novel end-to-end, prompt-based LLM framework for automated JUnit test generation. CANDOR orchestrates multiple specialized LLM agents to generate JUnit tests, including both high-quality test prefixes and accurate oracles. To mitigate the notorious hallucinations in LLMs, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generate accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments on the HumanEvalJava and LeetCodeJava datasets show that CANDOR can generate accurate oracles and is slightly better than EvoSuite in generating tests with high line coverage and clearly superior in terms of mutation score. Moreover, CANDOR significantly outperforms the state-of-the-art, prompt-based test generator LLM-Empirical, achieving improvements of 15.8 to 25.1 percentage points in oracle correctness on both correct and faulty source code. Ablation studies confirm the critical contributions of key agents in improving test prefix quality and oracle accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05109v5">A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 20 pages, 11 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Translating users' natural language queries (NL) into SQL queries (i.e., Text-to-SQL, a.k.a. NL2SQL) can significantly reduce barriers to accessing relational databases and support various commercial applications. The performance of Text-to-SQL has been greatly enhanced with the emergence of Large Language Models (LLMs). In this survey, we provide a comprehensive review of Text-to-SQL techniques powered by LLMs, covering its entire lifecycle from the following four aspects: (1) Model: Text-to-SQL translation techniques that tackle not only NL ambiguity and under-specification, but also properly map NL with database schema and instances; (2) Data: From the collection of training data, data synthesis due to training data scarcity, to Text-to-SQL benchmarks; (3) Evaluation: Evaluating Text-to-SQL methods from multiple angles using different metrics and granularities; and (4) Error Analysis: analyzing Text-to-SQL errors to find the root cause and guiding Text-to-SQL models to evolve. Moreover, we offer a rule of thumb for developing Text-to-SQL solutions. Finally, we discuss the research challenges and open problems of Text-to-SQL in the LLMs era.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09426v3">FlatQuant: Flatness Matters for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 27 pages, accepted to ICML 20205
    </div>
    <details class="paper-abstract">
      Recently, quantization has been widely used for the compression and acceleration of large language models (LLMs). Due to the outliers in LLMs, it is crucial to flatten weights and activations to minimize quantization error with equally spaced quantization points. Prior research explores various pre-quantization transformations to suppress outliers, such as per-channel scaling and Hadamard transformation. However, we observe that these transformed weights and activations can still exhibit steep and dispersed distributions. In this paper, we propose FlatQuant (Fast and Learnable Affine Transformation), a new post-training quantization approach that enhances the flatness of weights and activations. Our approach identifies optimal affine transformations for each linear layer, calibrated in hours via a lightweight objective. To reduce runtime overhead of affine transformation, we apply Kronecker product with two lightweight matrices, and fuse all operations in FlatQuant into a single kernel. Extensive experiments demonstrate that FlatQuant establishes a new state-of-the-art benchmark for quantization. For example, it achieves less than 1\% accuracy drop for W4A4 quantization on the LLaMA-3-70B model, surpassing SpinQuant by 7.5\%. Additionally, it provides up to 2.3x prefill speedup and 1.7x decoding speedup compared to the FP16 model. Code is available at: https://github.com/ruikangliu/FlatQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15557v2">MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn testing scenarios. However, multi-turn interaction is the common real-world usage of dialogue systems, yet testing methods for such interactions remain underexplored. This is largely due to the oracle problem in multi-turn testing, which continues to pose a significant challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a metamorphic multi-turn dialogue testing approach, which mitigates the test oracle problem in testing LLM-based dialogue systems. MORTAR formalises the multi-turn testing for dialogue systems, and automates the generation of question-answer dialogue test cases with multiple dialogue-level perturbations and metamorphic relations (MRs). The automated perturbation-MR matching mechanism allows MORTAR more flexibility and efficiency in metamorphic testing. The proposed approach is fully automated without reliance on potentially biased LLMs as test oracles. In testing six popular LLM-based dialogue systems, MORTAR reaches significantly better effectiveness with over 150\% more bugs revealed per test case when compared to the single-turn metamorphic testing baseline. On the quality of bugs, MORTAR reveals higher-quality bugs in terms of diversity, precision and uniqueness. MORTAR is expected to inspire more multi-turn testing approaches without LLM judges, and assist developers to evaluate the dialogue system performance more comprehensively with constrained test resources and budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08500v2">DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) is a commonly used approach for enhancing large language models (LLMs) with relevant and up-to-date information. However, the retrieved sources can often contain conflicting information and it remains unclear how models should address such discrepancies. In this work, we first propose a novel taxonomy of knowledge conflict types in RAG, along with the desired model behavior for each type. We then introduce CONFLICTS, a high-quality benchmark with expert annotations of conflict types in a realistic RAG setting. CONFLICTS is the first benchmark that enables tracking progress on how models address a wide range of knowledge conflicts. We conduct extensive experiments on this benchmark, showing that LLMs often struggle to appropriately resolve conflicts between sources. While prompting LLMs to explicitly reason about the potential conflict in the retrieved documents significantly improves the quality and appropriateness of their responses, substantial room for improvement in future research remains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12801v1">Mastering Da Vinci Code: A Comparative Study of Transformer, LLM, and PPO-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      The Da Vinci Code, a game of logical deduction and imperfect information, presents unique challenges for artificial intelligence, demanding nuanced reasoning beyond simple pattern recognition. This paper investigates the efficacy of various AI paradigms in mastering this game. We develop and evaluate three distinct agent architectures: a Transformer-based baseline model with limited historical context, several Large Language Model (LLM) agents (including Gemini, DeepSeek, and GPT variants) guided by structured prompts, and an agent based on Proximal Policy Optimization (PPO) employing a Transformer encoder for comprehensive game history processing. Performance is benchmarked against the baseline, with the PPO-based agent demonstrating superior win rates ($58.5\% \pm 1.0\%$), significantly outperforming the LLM counterparts. Our analysis highlights the strengths of deep reinforcement learning in policy refinement for complex deductive tasks, particularly in learning implicit strategies from self-play. We also examine the capabilities and inherent limitations of current LLMs in maintaining strict logical consistency and strategic depth over extended gameplay, despite sophisticated prompting. This study contributes to the broader understanding of AI in recreational games involving hidden information and multi-step logical reasoning, offering insights into effective agent design and the comparative advantages of different AI approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18583v3">PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The autoregressive nature of large language models (LLMs) limits inference speed. Each forward pass generates only a single token and is often bottlenecked by memory bandwidth. Speculative decoding alleviates this issue using a draft-then-verify approach to accelerate token generation. However, the overhead introduced during the draft phase and the training cost of the draft model limit the efficiency and adaptability of speculative decoding. In this work, we introduce PARallel Draft (PARD), a novel speculative decoding method that enables low-cost adaptation of autoregressive draft models into parallel draft models. PARD enhances inference efficiency by predicting multiple future tokens in a single forward pass of the draft phase, and incorporates a conditional drop token method to accelerate training. Its target-independence property allows a single draft model to be applied to an entire family of different models, minimizing the adaptation cost. Our proposed conditional drop token method can improves draft model training efficiency by 3x. On our optimized inference framework, PARD accelerates LLaMA3.1-8B inference by 4.08x, achieving 311.5 tokens per second.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10364v2">Can We Infer Confidential Properties of Training Data from LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05692v2">SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      The code generation capabilities of large language models(LLMs) have emerged as a critical dimension in evaluating their overall performance. However, prior research has largely overlooked the security risks inherent in the generated code. In this work, we introduce SafeGenBench, a benchmark specifically designed to assess the security of LLM-generated code. The dataset encompasses a wide range of common software development scenarios and vulnerability types. Building upon this benchmark, we develop an automatic evaluation framework that leverages both static application security testing(SAST) and LLM-based judging to assess the presence of security vulnerabilities in model-generated code. Through the empirical evaluation of state-of-the-art LLMs on SafeGenBench, we reveal notable deficiencies in their ability to produce vulnerability-free code. Our findings highlight pressing challenges and offer actionable insights for future advancements in the secure code generation performance of LLMs. The data and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12744v1">Rethinking Hate Speech Detection on Social Media: Can LLMs Replace Traditional Models?</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Hate speech detection across contemporary social media presents unique challenges due to linguistic diversity and the informal nature of online discourse. These challenges are further amplified in settings involving code-mixing, transliteration, and culturally nuanced expressions. While fine-tuned transformer models, such as BERT, have become standard for this task, we argue that recent large language models (LLMs) not only surpass them but also redefine the landscape of hate speech detection more broadly. To support this claim, we introduce IndoHateMix, a diverse, high-quality dataset capturing Hindi-English code-mixing and transliteration in the Indian context, providing a realistic benchmark to evaluate model robustness in complex multilingual scenarios where existing NLP methods often struggle. Our extensive experiments show that cutting-edge LLMs (such as LLaMA-3.1) consistently outperform task-specific BERT-based models, even when fine-tuned on significantly less data. With their superior generalization and adaptability, LLMs offer a transformative approach to mitigating online hate in diverse environments. This raises the question of whether future works should prioritize developing specialized models or focus on curating richer and more varied datasets to further enhance the effectiveness of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12728v1">MCTS-Refined CoT: High-Quality Fine-Tuning Data for LLM-Based Repository Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      LLMs demonstrate strong performance in auto-mated software engineering, particularly for code generation and issue resolution. While proprietary models like GPT-4o achieve high benchmarks scores on SWE-bench, their API dependence, cost, and privacy concerns limit adoption. Open-source alternatives offer transparency but underperform in complex tasks, especially sub-100B parameter models. Although quality Chain-of-Thought (CoT) data can enhance reasoning, current methods face two critical flaws: (1) weak rejection sampling reduces data quality, and (2) inadequate step validation causes error accumulation. These limitations lead to flawed reasoning chains that impair LLMs'ability to learn reliable issue resolution. The paper proposes MCTS-REFINE, an enhanced Monte Carlo Tree Search (MCTS)-based algorithm that dynamically validates and optimizes intermediate reasoning steps through a rigorous rejection sampling strategy, generating high-quality CoT data to improve LLM performance in issue resolution tasks. Key innovations include: (1) augmenting MCTS with a reflection mechanism that corrects errors via rejection sampling and refinement, (2) decomposing issue resolution into three subtasks-File Localization, Fault Localization, and Patch Generation-each with clear ground-truth criteria, and (3) enforcing a strict sampling protocol where intermediate outputs must exactly match verified developer patches, ensuring correctness across reasoning paths. Experiments on SWE-bench Lite and SWE-bench Verified demonstrate that LLMs fine-tuned with our CoT dataset achieve substantial improvements over baselines.Notably, Qwen2.5-72B- Instruct achieves 28.3%(Lite) and 35.0%(Verified) resolution rates, surpassing SOTA baseline SWE-Fixer-Qwen-72B with the same parameter scale, which only reached 24.7%(Lite) and 32.8%(Verified).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12713v1">Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition?</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(https://github.com/Humanity-s-Last-Code-Exam/HLCE).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12707v1">SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at https://aka.ms/SecurityLingua.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12691v1">Get on the Train or be Left on the Station: Using LLMs for Software Engineering Research</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 Accepted for publication at the 1st Workshop on Human-Centered AI for SE (Human AISE) held at the 33rd ACM International Conference on the Foundations of Software Engineering (FSE Companion '25), June 23-28, 2025, Trondheim, Norway
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) is not only transforming software engineering (SE) practice but is also poised to fundamentally disrupt how research is conducted in the field. While perspectives on this transformation range from viewing LLMs as mere productivity tools to considering them revolutionary forces, we argue that the SE research community must proactively engage with and shape the integration of LLMs into research practices, emphasizing human agency in this transformation. As LLMs rapidly become integral to SE research - both as tools that support investigations and as subjects of study - a human-centric perspective is essential. Ensuring human oversight and interpretability is necessary for upholding scientific rigor, fostering ethical responsibility, and driving advancements in the field. Drawing from discussions at the 2nd Copenhagen Symposium on Human-Centered AI in SE, this position paper employs McLuhan's Tetrad of Media Laws to analyze the impact of LLMs on SE research. Through this theoretical lens, we examine how LLMs enhance research capabilities through accelerated ideation and automated processes, make some traditional research practices obsolete, retrieve valuable aspects of historical research approaches, and risk reversal effects when taken to extremes. Our analysis reveals opportunities for innovation and potential pitfalls that require careful consideration. We conclude with a call to action for the SE research community to proactively harness the benefits of LLMs while developing frameworks and guidelines to mitigate their risks, to ensure continued rigor and impact of research in an AI-augmented future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12685v1">Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 10 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23884v5">Failure Modes of LLMs for Causal Reasoning on Narratives</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 ICML 2025 Workshop on Scaling up Intervention Models
    </div>
    <details class="paper-abstract">
      The ability to robustly identify causal relationships is essential for autonomous decision-making and adaptation to novel scenarios. However, accurately inferring causal structure requires integrating both world knowledge and abstract logical reasoning. In this work, we investigate the interaction between these two capabilities through the representative task of causal reasoning over narratives. Through controlled synthetic, semi-synthetic, and real-world experiments, we find that state-of-the-art large language models (LLMs) often rely on superficial heuristics -- for example, inferring causality from event order or recalling memorized world knowledge without attending to context. Furthermore, we show that simple reformulations of the task can elicit more robust reasoning behavior. Our evaluation spans a range of causal structures, from linear chains to complex graphs involving colliders and forks. These findings uncover systematic patterns in how LLMs perform causal reasoning and lay the groundwork for developing methods that better align LLM behavior with principled causal inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13820v1">Structured Program Synthesis using LLMs: Results and Insights from the IPARC Challenge</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      The IPARC Challenge, inspired by ARC, provides controlled program synthesis tasks over synthetic images to evaluate automatic program construction, focusing on sequence, selection, and iteration. This set of 600 tasks has resisted automated solutions. This paper presents a structured inductive programming approach with LLMs that successfully solves tasks across all IPARC categories. The controlled nature of IPARC reveals insights into LLM-based code generation, including the importance of prior structuring, LLMs' ability to aid structuring (requiring human refinement), the need to freeze correct code, the efficiency of code reuse, and how LLM-generated code can spark human creativity. These findings suggest valuable mechanisms for human-LLM collaboration in tackling complex program synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12365v1">Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      This survey paper outlines the key developments in the field of Large Language Models (LLMs), such as enhancing their reasoning skills, adaptability to various tasks, increased computational efficiency, and ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. They also manage to do more with less by applying scaling and optimization tricks for computing power conservation. This survey also offers a broader perspective on recent advancements in LLMs going beyond isolated aspects such as model architecture or ethical concerns. It categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. It also identifies underexplored areas such as interpretability, cross-modal integration and sustainability. With recent progress, challenges like huge computational costs, biases, and ethical risks remain constant. Addressing these requires bias mitigation, transparent decision-making, and clear ethical guidelines. Future research will focus on enhancing models ability to handle multiple input, thereby making them more intelligent, safe, and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09061v2">EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 4 figures, 7 pages, IEEE conference template
    </div>
    <details class="paper-abstract">
      This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12339v1">SheetMind: An End-to-End LLM-Powered Multi-Agent Framework for Spreadsheet Automation</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Ruiyan Zhu and Xi Cheng contributed equally to this work
    </div>
    <details class="paper-abstract">
      We present SheetMind, a modular multi-agent framework powered by large language models (LLMs) for spreadsheet automation via natural language instructions. The system comprises three specialized agents: a Manager Agent that decomposes complex user instructions into subtasks; an Action Agent that translates these into structured commands using a Backus Naur Form (BNF) grammar; and a Reflection Agent that validates alignment between generated actions and the user's original intent. Integrated into Google Sheets via a Workspace extension, SheetMind supports real-time interaction without requiring scripting or formula knowledge. Experiments on benchmark datasets demonstrate an 80 percent success rate on single step tasks and approximately 70 percent on multi step instructions, outperforming ablated and baseline variants. Our results highlight the effectiveness of multi agent decomposition and grammar based execution for bridging natural language and spreadsheet functionalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08127v3">Fino1: On the Transferability of Reasoning-Enhanced LLMs and Reinforcement Learning to Finance</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 13 pages, 2 figures, 3 Tables
    </div>
    <details class="paper-abstract">
      As the fundamental capability behind decision-making in finance, financial reasoning poses distinct challenges for LLMs. Although reinforcement learning (RL) have boosted generic reasoning, the progress in finance is hindered by the absence of empirical study of building effective financial chain-of-thought (CoT) corpus, a systematic comparison of different RL methods, and comprehensive benchmarks. To address these gaps, we introduce FinCoT, the first open high-fidelity CoT corpus for finance, distilled from seven QA datasets by a novel three-stage pipeline that incorporates domain supervision, iterative LLM refinement, and difficulty-aware filtering. Based on FinCoT, we develop Fin-o1, the first open financial reasoning models trained via supervised fine-tuning and GRPO-based RL. Our models outperform existing financial reasoning models and SOTA general models such as GPT-o1, DeepSeek-R1, and GPT-4.5. We also investigate the effectiveness of three different RL methods in improving domain-specific reasoning, offering the first such empirical study. We finally propose FinReason, the first financial reasoning benchmark covering multi-table analysis, long-context reasoning, and equation-based tasks, and evaluate 29 LLMs. Our extensive experiments reveal general reasoning models excel on standard benchmarks yet exhibit obvious performance degradation in financial contexts; even finance-tuned models like Dianjin-R1 and FinR1 degrade on lengthy documents. In contrast, our Fin-o1 models consistently outperform their backbones and larger GPT-o1 and DeepSeek-R1, confirming the effectiveness of our data building and model training strategy. Our study further shows that GRPO yields reliable gains whereas PPO and DPO do not, highlighting the need for targeted data and optimisation rather than scale alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12320v1">The Foundation Cracks: A Comprehensive Study on Bugs and Testing Practices in LLM Libraries</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) libraries have emerged as the foundational infrastructure powering today's AI revolution, serving as the backbone for LLM deployment, inference optimization, fine-tuning, and production serving across diverse applications. Despite their critical role in the LLM ecosystem, these libraries face frequent quality issues and bugs that threaten the reliability of AI systems built upon them. To address this knowledge gap, we present the first comprehensive empirical investigation into bug characteristics and testing practices in modern LLM libraries. We examine 313 bug-fixing commits extracted across two widely-adopted LLM libraries: HuggingFace Transformers and vLLM.Through rigorous manual analysis, we establish comprehensive taxonomies categorizing bug symptoms into 5 types and root causes into 14 distinct categories.Our primary discovery shows that API misuse has emerged as the predominant root cause (32.17%-48.19%), representing a notable transition from algorithm-focused defects in conventional deep learning frameworks toward interface-oriented problems. Additionally, we examine 7,748 test functions to identify 7 distinct test oracle categories employed in current testing approaches, with predefined expected outputs (such as specific tensors and text strings) being the most common strategy. Our assessment of existing testing effectiveness demonstrates that the majority of bugs escape detection due to inadequate test cases (41.73%), lack of test drivers (32.37%), and weak test oracles (25.90%). Drawing from these findings, we offer some recommendations for enhancing LLM library quality assurance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12307v1">Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Medical Question-Answering (QA) encompasses a broad spectrum of tasks, including multiple choice questions (MCQ), open-ended text generation, and complex computational reasoning. Despite this variety, a unified framework for delivering high-quality medical QA has yet to emerge. Although recent progress in reasoning-augmented large language models (LLMs) has shown promise, their ability to achieve comprehensive medical understanding is still largely unexplored. In this paper, we present Med-U1, a unified framework for robust reasoning across medical QA tasks with diverse output formats, ranging from MCQs to complex generation and computation tasks. Med-U1 employs pure large-scale reinforcement learning with mixed rule-based binary reward functions, incorporating a length penalty to manage output verbosity. With multi-objective reward optimization, Med-U1 directs LLMs to produce concise and verifiable reasoning chains. Empirical results reveal that Med-U1 significantly improves performance across multiple challenging Med-QA benchmarks, surpassing even larger specialized and proprietary models. Furthermore, Med-U1 demonstrates robust generalization to out-of-distribution (OOD) tasks. Extensive analysis presents insights into training strategies, reasoning chain length control, and reward design for medical LLMs. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05849v4">AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentVigil, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentVigil on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentVigil exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12299v1">QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Accept to ACLW 2025 (WOAH)
    </div>
    <details class="paper-abstract">
      The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12286v1">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce a diagnostic task: file path identification from issue descriptions alone, to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11027v2">Diversified Sampling Improves Scaling LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      While increasing training compute has significantly improved the performance of large language models (LLMs), similar gains have not been observed when scaling inference compute. We hypothesize that the primary issue lies in the uniformity of LLM outputs, which leads to inefficient sampling as models repeatedly generate similar but inaccurate responses. Motivated by an intriguing relationship between solution accuracy and response diversity, we propose DivSampling -- a novel and versatile sampling technique designed to enhance the diversity of candidate solutions by introducing prompt perturbations.DivSampling incorporates two categories of perturbations: task-agnostic approaches, which are general and not tailored to any specific task, and task-specific approaches, which are customized based on task content. Our theoretical analysis demonstrates that, under mild assumptions, the error rates of responses generated from diverse prompts are significantly lower compared to those produced by stationary prompts. Comprehensive evaluations across various tasks -- including reasoning, mathematics, and code generation -- highlight the effectiveness of DivSampling in improving solution accuracy. This scalable and efficient approach offers a new perspective on optimizing test-time inference, addressing limitations in current sampling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12618v1">OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Robust unlearning is crucial for safely deploying large language models (LLMs) in environments where data privacy, model safety, and regulatory compliance must be ensured. Yet the task is inherently challenging, partly due to difficulties in reliably measuring whether unlearning has truly occurred. Moreover, fragmentation in current methodologies and inconsistent evaluation metrics hinder comparative analysis and reproducibility. To unify and accelerate research efforts, we introduce OpenUnlearning, a standardized and extensible framework designed explicitly for benchmarking both LLM unlearning methods and metrics. OpenUnlearning integrates 9 unlearning algorithms and 16 diverse evaluations across 3 leading benchmarks (TOFU, MUSE, and WMDP) and also enables analyses of forgetting behaviors across 450+ checkpoints we publicly release. Leveraging OpenUnlearning, we propose a novel meta-evaluation benchmark focused specifically on assessing the faithfulness and robustness of evaluation metrics themselves. We also benchmark diverse unlearning methods and provide a comparative analysis against an extensive evaluation suite. Overall, we establish a clear, community-driven pathway toward rigorous development in LLM unlearning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12597v1">Automatic Expert Discovery in LLM Upcycling via Sparse Interpolated Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      We present Sparse Interpolated Mixture-of-Experts (SIMoE) instruction-tuning, an end-to-end algorithm designed to fine-tune a dense pre-trained Large Language Model (LLM) into a MoE-style model that possesses capabilities in multiple specialized domains. During instruction-tuning, SIMoE automatically identifies multiple specialized experts under a specified sparsity constraint, with each expert representing a structurally sparse subset of the seed LLM's parameters that correspond to domain-specific knowledge within the data. SIMoE simultaneously learns an input-dependent expert merging strategy via a router network, leveraging rich cross-expert knowledge for superior downstream generalization that surpasses existing baselines. Empirically, SIMoE consistently achieves state-of-the-art performance on common instruction-tuning benchmarks while maintaining an optimal performance-compute trade-off compared to all baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12577v1">OneEval: Benchmarking LLM Knowledge-intensive Reasoning over Diverse Knowledge Bases</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated substantial progress on reasoning tasks involving unstructured text, yet their capabilities significantly deteriorate when reasoning requires integrating structured external knowledge such as knowledge graphs, code snippets, or formal logic. This limitation is partly due to the absence of benchmarks capable of systematically evaluating LLM performance across diverse structured knowledge modalities. To address this gap, we introduce \textbf{\textsc{OneEval}}, a comprehensive benchmark explicitly designed to assess the knowledge-intensive reasoning capabilities of LLMs across four structured knowledge modalities, unstructured text, knowledge graphs, code, and formal logic, and five critical domains (general knowledge, government, science, law, and programming). \textsc{OneEval} comprises 4,019 carefully curated instances and includes a challenging subset, \textsc{OneEval}\textsubscript{Hard}, consisting of 1,285 particularly difficult cases. Through extensive evaluation of 18 state-of-the-art open-source and proprietary LLMs, we establish three core findings: a) \emph{persistent limitations in structured reasoning}, with even the strongest model achieving only 32.2\% accuracy on \textsc{OneEval}\textsubscript{Hard}; b) \emph{performance consistently declines as the structural complexity of the knowledge base increases}, with accuracy dropping sharply from 53\% (textual reasoning) to 25\% (formal logic); and c) \emph{diminishing returns from extended reasoning chains}, highlighting the critical need for models to adapt reasoning depth appropriately to task complexity. We release the \textsc{OneEval} datasets, evaluation scripts, and baseline results publicly, accompanied by a leaderboard to facilitate ongoing advancements in structured knowledge reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02678v3">TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11242v4">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      This study examines the growing use of Large Language Models (LLMs) in child-centered applications, highlighting safety and ethical concerns such as bias, harmful content, and cultural insensitivity. Despite their potential to enhance learning, there is a lack of standardized frameworks to mitigate these risks. Through a systematic literature review, we identify key parental and empirical concerns, including toxicity and ethical breaches in AI outputs. Moreover, to address these issues, this paper proposes a protection framework for safe Child-LLM interaction, incorporating metrics for content safety, behavioral ethics, and cultural sensitivity. The framework provides practical tools for evaluating LLM safety, offering guidance for developers, policymakers, and educators to ensure responsible AI deployment for children.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08001v2">Reparameterized LLM Training via Orthogonal Equivalence Transformation</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Technical report v2 (37 pages, 24 figures, project page: https://spherelab.ai/poet/)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15513v3">PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Accepted by ACL2025 Main, a sibling project to SafeRLHF and BeaverTails
    </div>
    <details class="paper-abstract">
      In this study, we introduce the safety human preference dataset, PKU-SafeRLHF, designed to promote research on safety alignment in large language models (LLMs). As a sibling project to SafeRLHF and BeaverTails, we separate annotations of helpfulness and harmlessness for question-answering pairs, providing distinct perspectives on these coupled attributes. Overall, we provide 44.6k refined prompts and 265k question-answer pairs with safety meta-labels for 19 harm categories and three severity levels ranging from minor to severe, with answers generated by Llama-family models. Based on this, we collected 166.8k preference data, including dual-preference (helpfulness and harmlessness decoupled) and single-preference data (trade-off the helpfulness and harmlessness from scratch), respectively. Using the large-scale annotation data, we further train severity-sensitive moderation for the risk control of LLMs and safety-centric RLHF algorithms for the safety alignment of LLMs. We believe this dataset will be a valuable resource for the community, aiding in the safe deployment of LLMs. Data is available at https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12552v1">Profiling News Media for Factuality and Bias Using LLMs and the Fact-Checking Methodology of Human Experts</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Accepted to Findings of the Association for Computational Linguistics (ACL) 2025
    </div>
    <details class="paper-abstract">
      In an age characterized by the proliferation of mis- and disinformation online, it is critical to empower readers to understand the content they are reading. Important efforts in this direction rely on manual or automatic fact-checking, which can be challenging for emerging claims with limited information. Such scenarios can be handled by assessing the reliability and the political bias of the source of the claim, i.e., characterizing entire news outlets rather than individual claims or articles. This is an important but understudied research direction. While prior work has looked into linguistic and social contexts, we do not analyze individual articles or information in social media. Instead, we propose a novel methodology that emulates the criteria that professional fact-checkers use to assess the factuality and political bias of an entire outlet. Specifically, we design a variety of prompts based on these criteria and elicit responses from large language models (LLMs), which we aggregate to make predictions. In addition to demonstrating sizable improvements over strong baselines via extensive experiments with multiple LLMs, we provide an in-depth error analysis of the effect of media popularity and region on model performance. Further, we conduct an ablation study to highlight the key components of our dataset that contribute to these improvements. To facilitate future research, we released our dataset and code at https://github.com/mbzuai-nlp/llm-media-profiling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10949v2">Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Current LLM safety defenses fail under decomposition attacks, where a malicious goal is decomposed into benign subtasks that circumvent refusals. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent, leaving them blind to malicious intent that emerges over a sequence of seemingly benign instructions. We therefore propose adding an external monitor that observes the conversation at a higher granularity. To facilitate our study of monitoring decomposition attacks, we curate the largest and most diverse dataset to date, including question-answering, text-to-image, and agentic tasks. We verify our datasets by testing them on frontier LLMs and show an 87% attack success rate on average on GPT-4o. This confirms that decomposition attack is broadly effective. Additionally, we find that random tasks can be injected into the decomposed subtasks to further obfuscate malicious intents. To defend in real time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each subtask. We show that a carefully prompt engineered lightweight monitor achieves a 93% defense success rate, beating reasoning models like o3 mini as a monitor. Moreover, it remains robust against random task injection and cuts cost by 90% and latency by 50%. Our findings suggest that lightweight sequential monitors are highly effective in mitigating decomposition attacks and are viable in deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12509v1">Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Verifying the reliability of complex, multi-step reasoning in Large Language Models (LLMs) remains a fundamental challenge, as existing methods often lack both faithfulness and precision. To address this issue, we propose the Graph of Verification (GoV) framework. GoV offers three key contributions: First, it explicitly models the underlying deductive process as a directed acyclic graph (DAG), whether this structure is implicit or explicitly constructed. Second, it enforces a topological order over the DAG to guide stepwise verification. Third, GoV introduces the notion of customizable node blocks, which flexibly define the verification granularity, from atomic propositions to full paragraphs, while ensuring that all requisite premises derived from the graph are provided as contextual input for each verification unit. We evaluate GoV on the Number Triangle Summation task and the ProcessBench benchmark with varying levels of reasoning complexity. Experimental results show that GoV substantially improves verification accuracy, faithfulness, and error localization when compared to conventional end-to-end verification approaches. Our code and data are available at https://github.com/Frevor/Graph-of-Verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04223v2">VideoQA in the Era of LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 IJCV'25
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) are flourishing and has advanced many video-language tasks. As a golden testbed, Video Question Answering (VideoQA) plays pivotal role in Video-LLM developing. This work conducts a timely and comprehensive study of Video-LLMs' behavior in VideoQA, aiming to elucidate their success and failure modes, and provide insights towards more human-like video understanding and question answering. Our analyses demonstrate that Video-LLMs excel in VideoQA; they can correlate contextual cues and generate plausible responses to questions about varied video contents. However, models falter in handling video temporality, both in reasoning about temporal content ordering and grounding QA-relevant temporal moments. Moreover, the models behave unintuitively - they are unresponsive to adversarial video perturbations while being sensitive to simple variations of candidate answers and questions. Also, they do not necessarily generalize better. The findings demonstrate Video-LLMs' QA capability in standard condition yet highlight their severe deficiency in robustness and interpretability, suggesting the urgent need on rationales in Video-LLM developing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12473v1">TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 ACL 2025, 26 pages, 13 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Model routing allocates queries to the suitable model, improving system performance while reducing costs. However, existing routing methods face practical limitations that hinder scalability in large-scale applications and struggle to keep up with the rapid growth of the large language model (LLM) ecosystem. To tackle these challenges, we propose TagRouter, a training-free model routing method designed to optimize the synergy among multiple LLMs for open-domain text generation tasks. Experimental results demonstrate that TagRouter outperforms 13 baseline methods, increasing the accept rate of system by 6.15% and reducing costs by 17.20%, achieving optimal cost-efficiency. Our findings provides the LLM community with an efficient and scalable solution for model ensembling, offering users an evolvable "super model."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01668v2">CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Accepted as Main of ACL2025
    </div>
    <details class="paper-abstract">
      Current inference scaling methods, such as Self-consistency and Best-of-N, have proven effective in improving the accuracy of LLMs on complex reasoning tasks. However, these methods rely heavily on the quality of candidate responses and are unable to produce correct answers when all candidates are incorrect. In this paper, we propose a novel inference scaling strategy, CoT-based Synthesizer, which leverages CoT reasoning to synthesize superior answers by analyzing complementary information from multiple candidate responses, even when all candidate responses are flawed. To enable a lightweight and cost-effective implementation, we introduce an automated data generation pipeline that creates diverse training data. This allows smaller LLMs trained on this data to improve the inference accuracy of larger models, including API-based LLMs. Experimental results across four benchmark datasets with seven policy models demonstrate that our method significantly enhances performance, with gains of 11.8% for Llama3-8B and 10.3% for GPT-4o on the MATH dataset. The corresponding training data and code are publicly available on https://github.com/RUCKBReasoning/CoT-based-Synthesizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12421v1">Plan Your Travel and Travel with Your Plan: Wide-Horizon Planning and Evaluation via LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Travel planning is a complex task requiring the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints and preferences in the context, leading to suboptimal itineraries. We formulate this as an $L^3$ planning problem, emphasizing long context, long instruction, and long output. To tackle this, we introduce Multiple Aspects of Planning (MAoP), enabling LLMs to conduct wide-horizon thinking to solve complex planning problems. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planning models, enabling strong inference-time scalability for better performance. In addition, current benchmarks overlook travel's dynamic nature, where past events impact subsequent journeys, failing to reflect real-world feasibility. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world travel simulation. This work advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through agent-based simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12379v1">Training-free LLM Merging for Multi-task Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. In this paper, we explore an important question: is it possible to combine these specialized models to create a unified model with multi-task capabilities. We introduces Hierarchical Iterative Merging (Hi-Merging), a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts. Extensive experiments on multiple-choice and question-answering tasks in both Chinese and English validate Hi-Merging's ability for multi-task learning. The results demonstrate that Hi-Merging consistently outperforms existing merging techniques and surpasses the performance of models fine-tuned on combined datasets in most scenarios. Code is available at: https://github.com/Applied-Machine-Learning-Lab/Hi-Merging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12376v1">ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Accepted at ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: https://github.com/ulab-uiuc/consistencychecker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03213v2">ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely deployed in a variety of applications, and the context length is rapidly increasing to handle tasks such as long-document QA and complex logical reasoning. However, long context poses significant challenges for inference efficiency, including high memory costs of key-value (KV) cache and increased latency due to extensive memory accesses. Recent works have proposed compressing KV cache to approximate computation, but these methods either evict tokens permanently, never recalling them for later inference, or recall previous tokens at the granularity of pages divided by textual positions. Both approaches degrade the model accuracy and output quality. To achieve efficient and accurate recallable KV cache compression, we introduce ClusterKV, which recalls tokens at the granularity of semantic clusters. We design and implement efficient algorithms and systems for clustering, selection, indexing and caching. Experiment results show that ClusterKV attains negligible accuracy loss across various tasks with 32k context lengths, using only a 1k to 2k KV cache budget, and achieves up to a 2$\times$ speedup in latency and a 2.5$\times$ improvement in decoding throughput. Compared to SoTA recallable KV compression methods, ClusterKV demonstrates higher model accuracy and output quality, while maintaining or exceeding inference efficiency. Our code is available at https://github.com/sjtu-zhao-lab/ClusterKV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11934v5">Stepwise Reasoning Error Disruption Attack of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications. Our code is available at: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18415v2">BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Efficient deployment of 1-bit Large Language Models (LLMs) is hindered by activation outliers, which complicate quantization to low bit-widths. We introduce BitNet v2, a novel framework enabling native 4-bit activation quantization for 1-bit LLMs. To tackle outliers in attention and feed-forward network activations, we propose H-BitLinear, a module applying an online Hadamard transformation prior to activation quantization. This transformation smooths sharp activation distributions into more Gaussian-like forms, suitable for low-bit representation. Experiments show BitNet v2 trained from scratch with 8-bit activations matches BitNet b1.58 performance. Crucially, BitNet v2 achieves minimal performance degradation when trained with native 4-bit activations, significantly reducing memory footprint and computational cost for batched inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04951v3">Unsafe LLM-Based Search: Quantitative Analysis and Mitigation of Safety Risks in AI Web Search</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have significantly enhanced the capabilities of AI-Powered Search Engines (AIPSEs), offering precise and efficient responses by integrating external databases with pre-existing knowledge. However, we observe that these AIPSEs raise risks such as quoting malicious content or citing malicious websites, leading to harmful or unverified information dissemination. In this study, we conduct the first safety risk quantification on seven production AIPSEs by systematically defining the threat model, risk type, and evaluating responses to various query types. With data collected from PhishTank, ThreatBook, and LevelBlue, our findings reveal that AIPSEs frequently generate harmful content that contains malicious URLs even with benign queries (e.g., with benign keywords). We also observe that directly querying a URL will increase the number of main risk-inclusive responses, while querying with natural language will slightly mitigate such risk. Compared to traditional search engines, AIPSEs outperform in both utility and safety. We further perform two case studies on online document spoofing and phishing to show the ease of deceiving AIPSEs in the real-world setting. To mitigate these risks, we develop an agent-based defense with a GPT-4.1-based content refinement tool and a URL detector. Our evaluation shows that our defense can effectively reduce the risk, with only a minor cost of reducing available information by approximately 10.7%. Our research highlights the urgent need for robust safety measures in AIPSEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04125v3">VulScribeR: Exploring RAG-based Vulnerability Augmentation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 25 pages, 6 figures, 8 tables, 3 prompt templates, 1 algorithm
    </div>
    <details class="paper-abstract">
      Detecting vulnerabilities is vital for software security, yet deep learning-based vulnerability detectors (DLVD) face a data shortage, which limits their effectiveness. Data augmentation can potentially alleviate the data shortage, but augmenting vulnerable code is challenging and requires a generative solution that maintains vulnerability. Previous works have only focused on generating samples that contain single statements or specific types of vulnerabilities. Recently, large language models (LLMs) have been used to solve various code generation and comprehension tasks with inspiring results, especially when fused with retrieval augmented generation (RAG). Therefore, we propose VulScribeR, a novel LLM-based solution that leverages carefully curated prompt templates to augment vulnerable datasets. More specifically, we explore three strategies to augment both single and multi-statement vulnerabilities, with LLMs, namely Mutation, Injection, and Extension. Our extensive evaluation across four vulnerability datasets and DLVD models, using three LLMs, show that our approach beats two SOTA methods Vulgen and VGX, and Random Oversampling (ROS) by 27.48%, 27.93%, and 15.41% in f1-score with 5K generated vulnerable samples on average, and 53.84%, 54.10%, 69.90%, and 40.93% with 15K generated vulnerable samples. Our approach demonstrates its feasibility for large-scale data augmentation by generating 1K samples at as cheap as US$ 1.88.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11418v1">Efficient Long-Context LLM Inference via KV Cache Clustering</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with extended context windows have become increasingly prevalent for tackling complex tasks. However, the substantial Key-Value (KV) cache required for long-context LLMs poses significant deployment challenges. Existing approaches either discard potentially critical information needed for future generations or offer limited efficiency gains due to high computational overhead. In this paper, we introduce Chelsea, a simple yet effective framework for online KV cache clustering. Our approach is based on the observation that key states exhibit high similarity along the sequence dimension. To enable efficient clustering, we divide the sequence into chunks and propose Chunked Soft Matching, which employs an alternating partition strategy within each chunk and identifies clusters based on similarity. Chelsea then merges the KV cache within each cluster into a single centroid. Additionally, we provide a theoretical analysis of the computational complexity and the optimality of the intra-chunk partitioning strategy. Extensive experiments across various models and long-context benchmarks demonstrate that Chelsea achieves up to 80% reduction in KV cache memory usage while maintaining comparable model performance. Moreover, with minimal computational overhead, Chelsea accelerates the decoding stage of inference by up to 3.19$\times$ and reduces end-to-end latency by up to 2.72$\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12278v1">Can LLMs Generate High-Quality Test Cases for Algorithm Problems? TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      We introduce TestCase-Eval, a new benchmark for systematic evaluation of LLMs in test-case generation. TestCase-Eval includes 500 algorithm problems and 100,000 human-crafted solutions from the Codeforces platform. It focuses on two pivotal tasks: (1) Fault Coverage, which measures how well LLM-generated test sets probe diverse input scenarios and cover a wide range of potential failure modes. (2) Fault Exposure, which evaluates whether LLMs can craft a tailored test input that reveals a specific incorrect code implementation. We provide a comprehensive assessment of 19 state-of-the-art open-source and proprietary LLMs on TestCase-Eval, offering insights into their strengths and limitations in generating effective test cases for algorithm problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20965v2">AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 ICLR 2025 Workshop BuildingTrust
    </div>
    <details class="paper-abstract">
      We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12266v1">The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 ACL 2025; 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have significantly impacted Task-Oriented Dialog Systems (TODS) but continue to face notable performance challenges, especially in zero-shot scenarios. While prior work has noted this performance gap, the behavioral factors driving the performance gap remain under-explored. This study proposes a comprehensive evaluation framework to quantify the behavior gap between AI agents and human experts, focusing on discrepancies in dialog acts, tool usage, and knowledge utilization. Our findings reveal that this behavior gap is a critical factor negatively impacting the performance of LLM agents. Notably, as task complexity increases, the behavior gap widens (correlation: 0.963), leading to a degradation of agent performance on complex task-oriented dialogs. For the most complex task in our study, even the GPT-4o-based agent exhibits low alignment with human behavior, with low F1 scores for dialog acts (0.464), excessive and often misaligned tool usage with a F1 score of 0.139, and ineffective usage of external knowledge. Reducing such behavior gaps leads to significant performance improvement (24.3% on average). This study highlights the importance of comprehensive behavioral evaluations and improved alignment strategies to enhance the effectiveness of LLM-based TODS in handling complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07087v2">Applying Cognitive Design Patterns to General LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 10 pages + references, 2 figures, 3 tables. Accepted for oral presentation at AGI25
    </div>
    <details class="paper-abstract">
      One goal of AI (and AGI) is to identify and understand specific mechanisms and representations sufficient for general intelligence. Often, this work manifests in research focused on architectures and many cognitive architectures have been explored in AI/AGI. However, different research groups and even different research traditions have somewhat independently identified similar/common patterns of processes and representations or "cognitive design patterns" that are manifest in existing architectures. Today, AI systems exploiting large language models (LLMs) offer a relatively new combination of mechanisms and representations available for exploring the possibilities of general intelligence. This paper outlines a few recurring cognitive design patterns that have appeared in various pre-transformer AI architectures. We then explore how these patterns are evident in systems using LLMs, especially for reasoning and interactive ("agentic") use cases. Examining and applying these recurring patterns enables predictions of gaps or deficiencies in today's Agentic LLM Systems and identification of subjects of future research towards general intelligence using generative foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12240v1">Mind the XAI Gap: A Human-Centered LLM Framework for Democratizing Explainable AI</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted for publication at The 3rd World Conference on eXplainable Artificial Intelligence. This version corresponds to the camera-ready manuscript submitted to the conference proceedings
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is rapidly embedded in critical decision-making systems, however their foundational ``black-box'' models require eXplainable AI (XAI) solutions to enhance transparency, which are mostly oriented to experts, making no sense to non-experts. Alarming evidence about AI's unprecedented human values risks brings forward the imperative need for transparent human-centered XAI solutions. In this work, we introduce a domain-, model-, explanation-agnostic, generalizable and reproducible framework that ensures both transparency and human-centered explanations tailored to the needs of both experts and non-experts. The framework leverages Large Language Models (LLMs) and employs in-context learning to convey domain- and explainability-relevant contextual knowledge into LLMs. Through its structured prompt and system setting, our framework encapsulates in one response explanations understandable by non-experts and technical information to experts, all grounded in domain and explainability principles. To demonstrate the effectiveness of our framework, we establish a ground-truth contextual ``thesaurus'' through a rigorous benchmarking with over 40 data, model, and XAI combinations for an explainable clustering analysis of a well-being scenario. Through a comprehensive quality and human-friendliness evaluation of our framework's explanations, we prove high content quality through strong correlations with ground-truth explanations (Spearman rank correlation=0.92) and improved interpretability and human-friendliness to non-experts through a user study (N=56). Our overall evaluation confirms trust in LLMs as HCXAI enablers, as our framework bridges the above Gaps by delivering (i) high-quality technical explanations aligned with foundational XAI methods and (ii) clear, efficient, and interpretable human-centered explanations for non-experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12227v1">Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Submitted to AIES Conference
    </div>
    <details class="paper-abstract">
      Causal discovery (CD) plays a pivotal role in understanding the mechanisms underlying complex systems. While recent algorithms can detect spurious associations and latent confounding, many struggle to recover fairness-relevant pathways in realistic, noisy settings. Large Language Models (LLMs), with their access to broad semantic knowledge, offer a promising complement to statistical CD approaches, particularly in domains where metadata provides meaningful relational cues. Ensuring fairness in machine learning requires understanding how sensitive attributes causally influence outcomes, yet CD methods often introduce spurious or biased pathways. We propose a hybrid LLM-based framework for CD that extends a breadth-first search (BFS) strategy with active learning and dynamic scoring. Variable pairs are prioritized for LLM-based querying using a composite score based on mutual information, partial correlation, and LLM confidence, improving discovery efficiency and robustness. To evaluate fairness sensitivity, we construct a semi-synthetic benchmark from the UCI Adult dataset, embedding a domain-informed causal graph with injected noise, label corruption, and latent confounding. We assess how well CD methods recover both global structure and fairness-critical paths. Our results show that LLM-guided methods, including the proposed method, demonstrate competitive or superior performance in recovering such pathways under noisy conditions. We highlight when dynamic scoring and active querying are most beneficial and discuss implications for bias auditing in real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v2">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12204v1">Semantic Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 18 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Conventional operating system scheduling algorithms are largely content-ignorant, making decisions based on factors such as latency or fairness without considering the actual intents or semantics of processes. Consequently, these algorithms often do not prioritize tasks that require urgent attention or carry higher importance, such as in emergency management scenarios. However, recent advances in language models enable semantic analysis of processes, allowing for more intelligent and context-aware scheduling decisions. In this paper, we introduce the concept of semantic scheduling in scheduling of requests from large language models (LLM), where the semantics of the process guide the scheduling priorities. We present a novel scheduling algorithm with optimal time complexity, designed to minimize the overall waiting time in LLM-based prompt scheduling. To illustrate its effectiveness, we present a medical emergency management application, underscoring the potential benefits of semantic scheduling for critical, time-sensitive tasks. The code and data are available at https://github.com/Wenyueh/latency_optimization_with_priority_constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12202v1">A Fast, Reliable, and Secure Programming Language for LLM Agents with Code Actions</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) are often deployed as agents, calling external tools adaptively to solve tasks. Rather than directly calling tools, it can be more effective for LLMs to write code to perform the tool calls, enabling them to automatically generate complex control flow such as conditionals and loops. Such code actions are typically provided as Python code, since LLMs are quite proficient at it; however, Python may not be the ideal language due to limited built-in support for performance, security, and reliability. We propose a novel programming language for code actions, called Quasar, which has several benefits: (1) automated parallelization to improve performance, (2) uncertainty quantification to improve reliability and mitigate hallucinations, and (3) security features enabling the user to validate actions. LLMs can write code in a subset of Python, which is automatically transpiled to Quasar. We evaluate our approach on the ViperGPT visual question answering agent, applied to the GQA dataset, demonstrating that LLMs with Quasar actions instead of Python actions retain strong performance, while reducing execution time when possible by 42%, improving security by reducing user approval interactions when possible by 52%, and improving reliability by applying conformal prediction to achieve a desired target coverage level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12182v1">Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted by 2025 International Conference on Artificial Intelligence, Human-Computer Interaction and Natural Language Processing
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential in medical question answering (MedQA), yet adapting them to biomedical reasoning remains challenging due to domain-specific complexity and limited supervision. In this work, we study how prompt design and lightweight fine-tuning affect the performance of open-source LLMs on PubMedQA, a benchmark for multiple-choice biomedical questions. We focus on two widely used prompting strategies - standard instruction prompts and Chain-of-Thought (CoT) prompts - and apply QLoRA for parameter-efficient instruction tuning. Across multiple model families and sizes, our experiments show that CoT prompting alone can improve reasoning in zero-shot settings, while instruction tuning significantly boosts accuracy. However, fine-tuning on CoT prompts does not universally enhance performance and may even degrade it for certain larger models. These findings suggest that reasoning-aware prompts are useful, but their benefits are model- and scale-dependent. Our study offers practical insights into combining prompt engineering with efficient finetuning for medical QA applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12158v1">A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12152v1">Because we have LLMs, we Can and Should Pursue Agentic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      The era of Large Language Models (LLMs) presents a new opportunity for interpretability--agentic interpretability: a multi-turn conversation with an LLM wherein the LLM proactively assists human understanding by developing and leveraging a mental model of the user, which in turn enables humans to develop better mental models of the LLM. Such conversation is a new capability that traditional `inspective' interpretability methods (opening the black-box) do not use. Having a language model that aims to teach and explain--beyond just knowing how to talk--is similar to a teacher whose goal is to teach well, understanding that their success will be measured by the student's comprehension. While agentic interpretability may trade off completeness for interactivity, making it less suitable for high-stakes safety situations with potentially deceptive models, it leverages a cooperative model to discover potentially superhuman concepts that can improve humans' mental model of machines. Agentic interpretability introduces challenges, particularly in evaluation, due to what we call `human-entangled-in-the-loop' nature (humans responses are integral part of the algorithm), making the design and evaluation difficult. We discuss possible solutions and proxy goals. As LLMs approach human parity in many tasks, agentic interpretability's promise is to help humans learn the potentially superhuman concepts of the LLMs, rather than see us fall increasingly far from understanding them.
    </details>
</div>
